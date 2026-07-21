// gds_direct_bench.cpp -- DIRECT (GDS/cuFile, HBM->NVMe p2p DMA) vs STAGED
// (HBM->pinned host->O_DIRECT pwrite) checkpoint-leg microbench, with CPU-energy
// (RAPL sysfs delta, wrap-aware) per leg. The measured "direct" path the model
// currently projects (~53% NVMe dump saving from the staged decomposition).
//
// Build (needs GDS stack: cufile.h + libcufile; probe with h100_probe.sh first):
//   nvcc -O2 -o gds_bench scripts/gds_direct_bench.cpp -lcufile
// Run (root; file on the NVMe scratch):
//   ./gds_bench --gb 16 --file /scratch/gds.bin --mode both
// Output: one line per leg: mode,dir,GB,seconds,GBps,cpu_J  (parse-friendly).
#include <cuda_runtime.h>
#include <cufile.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <glob.h>
#include <string>
#include <vector>

#define CK(x) do { auto _e=(x); if (_e!=cudaSuccess) { \
  fprintf(stderr,"CUDA err %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(_e)); exit(1);} } while(0)

// ---- RAPL package energy (sum of top-level domains), wrap-aware ----
struct Rapl {
  std::vector<std::string> dirs; std::vector<long long> maxr;
  Rapl() {
    glob_t g{}; glob("/sys/class/powercap/intel-rapl:*", 0, nullptr, &g);
    for (size_t i=0;i<g.gl_pathc;i++) {
      std::string d=g.gl_pathv[i];
      if (d.find(':')==d.rfind(':')) {              // top-level only (one ':')
        dirs.push_back(d);
        long long m=0; FILE*f=fopen((d+"/max_energy_range_uj").c_str(),"r");
        if (f){ if(fscanf(f,"%lld",&m)!=1) m=0; fclose(f);} maxr.push_back(m);
      }
    }
    globfree(&g);
  }
  double read() {                                    // Joules, monotonic-ish (single read)
    double t=0; for (auto&d:dirs){ long long v=0;
      FILE*f=fopen((d+"/energy_uj").c_str(),"r");
      if(f){ if(fscanf(f,"%lld",&v)!=1) v=0; fclose(f);} t+=v/1e6; } return t;
  }
  double delta(double a, double b) {                 // b-a with at most one wrap per domain
    if (b>=a) return b-a;
    double m=0; for (auto r:maxr) m+=r/1e6; return b-a+m;
  }
};

static double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static void report(const char*mode,const char*dir,double gb,double s,double j){
  printf("RESULT,%s,%s,%.2f,%.3f,%.3f,%.1f\n",mode,dir,gb,s,gb/s,j); fflush(stdout);
}

int main(int argc,char**argv){
  double gb=8; const char*path="/tmp/gds.bin"; std::string mode="both"; size_t chunk=256ull<<20;
  for(int i=1;i<argc;i++){ std::string a=argv[i];
    if(a=="--gb")gb=atof(argv[++i]); else if(a=="--file")path=argv[++i];
    else if(a=="--mode")mode=argv[++i]; else if(a=="--chunk-mb")chunk=(size_t)atol(argv[++i])<<20; }
  size_t total=(size_t)(gb*(1ull<<30)); total-= total%4096;       // O_DIRECT alignment
  Rapl rapl;
  void* dbuf=nullptr; CK(cudaMalloc(&dbuf,total)); CK(cudaMemset(dbuf,0xA5,total));
  CK(cudaDeviceSynchronize());
  printf("# gds_bench gb=%.2f file=%s mode=%s chunk=%zuMB rapl_domains=%zu\n",
         gb,path,mode.c_str(),chunk>>20,rapl.dirs.size());

  if(mode=="direct"||mode=="both"){
    CUfileError_t st=cuFileDriverOpen();
    if(st.err!=CU_FILE_SUCCESS){ fprintf(stderr,"cuFileDriverOpen failed (%d) -- no GDS; staged only\n",st.err); }
    else{
      st=cuFileBufRegister(dbuf,total,0);
      if(st.err!=CU_FILE_SUCCESS) fprintf(stderr,"warn: cuFileBufRegister %d (continuing unregistered)\n",st.err);
      for(const char*dir:{"write","read"}){
        int flags= strcmp(dir,"write")==0 ? (O_CREAT|O_WRONLY|O_DIRECT) : (O_RDONLY|O_DIRECT);
        int fd=open(path,flags,0644); if(fd<0){perror("open");exit(1);}
        CUfileDescr_t cf{}; cf.handle.fd=fd; cf.type=CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        CUfileHandle_t fh; st=cuFileHandleRegister(&fh,&cf);
        if(st.err!=CU_FILE_SUCCESS){fprintf(stderr,"cuFileHandleRegister %d\n",st.err);exit(1);}
        double e0=rapl.read(), t0=now();
        for(size_t off=0;off<total;off+=chunk){
          size_t n=std::min(chunk,total-off);
          ssize_t r= strcmp(dir,"write")==0
            ? cuFileWrite(fh,dbuf,n,(off_t)off,(off_t)off)
            : cuFileRead (fh,dbuf,n,(off_t)off,(off_t)off);
          if(r<0||(size_t)r!=n){fprintf(stderr,"cuFile%s %zd @%zu\n",dir,r,off);exit(1);}
        }
        if(strcmp(dir,"write")==0) fsync(fd);
        double t1=now(), e1=rapl.read();
        report("direct",dir,total/1e9,t1-t0,rapl.delta(e0,e1));
        cuFileHandleDeregister(fh); close(fd);
      }
      cuFileBufDeregister(dbuf); cuFileDriverClose();
    }
  }

  if(mode=="staged"||mode=="both"){
    void* hbuf=nullptr;
    if(posix_memalign(&hbuf,4096,chunk)){perror("memalign");exit(1);}
    CK(cudaHostRegister(hbuf,chunk,cudaHostRegisterDefault));       // pinned -> full PCIe rate
    for(const char*dir:{"write","read"}){
      int flags= strcmp(dir,"write")==0 ? (O_CREAT|O_WRONLY|O_DIRECT) : (O_RDONLY|O_DIRECT);
      int fd=open(path,flags,0644); if(fd<0){perror("open");exit(1);}
      double e0=rapl.read(), t0=now();
      for(size_t off=0;off<total;off+=chunk){
        size_t n=std::min(chunk,total-off);
        if(strcmp(dir,"write")==0){
          CK(cudaMemcpy(hbuf,(char*)dbuf+off,n,cudaMemcpyDeviceToHost));
          if(pwrite(fd,hbuf,n,(off_t)off)!=(ssize_t)n){perror("pwrite");exit(1);}
        }else{
          if(pread(fd,hbuf,n,(off_t)off)!=(ssize_t)n){perror("pread");exit(1);}
          CK(cudaMemcpy((char*)dbuf+off,hbuf,n,cudaMemcpyHostToDevice));
        }
      }
      if(strcmp(dir,"write")==0) fsync(fd);
      double t1=now(), e1=rapl.read();
      report("staged",dir,total/1e9,t1-t0,rapl.delta(e0,e1));
      close(fd);
    }
    CK(cudaHostUnregister(hbuf)); free(hbuf);
  }
  cudaFree(dbuf); unlink(path);
  return 0;
}
