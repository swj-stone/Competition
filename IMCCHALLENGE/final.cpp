#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("O3,unroll-loops")
#endif
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
using namespace std;
vector<char> EY(){vector<char> K3;
K3.reserve(1 <<27);
char HG[1 <<16];
size_t n;
while((n=fread(HG,1,sizeof(HG),stdin))> 0)
K3.insert(K3.end(),HG,HG+n);
K3.push_back('\0');
return K3;}
int NV,NF,B0,EX=-1,EW,EU,EV,ET;
vector<double> Gs[4],GI[4];
vector<double> X,Y,Z;
vector<double> oX,oY,oZ;
vector<int> FA,FB,FC;
vector<int> L3,L2,L1;
void Hl(){vector<char> K3=EY();
char*p=K3.data();
long nv=strtol(p,&p,10);
long nf=strtol(p,&p,10);
NV=nv;NF=nf;
X.resize(nv);Y.resize(nv);Z.resize(nv);
FA.resize(nf);FB.resize(nf);FC.resize(nf);
for(long i=0;i <nv;++i){while(*p==' '||*p=='\n'||*p=='\r'||*p=='\t')++p;
++p;
X[i]=strtod(p,&p);Y[i]=strtod(p,&p);Z[i]=strtod(p,&p);}
for(long i=0;i <nf;++i){while(*p==' '||*p=='\n'||*p=='\r'||*p=='\t')++p;
++p;
FA[i]=(int)strtol(p,&p,10)-1;
FB[i]=(int)strtol(p,&p,10)-1;
FC[i]=(int)strtol(p,&p,10)-1;}
L3=FA;L2=FB;L1=FC;}
vector<array<double,10>> Q;
vector<vector<int>> vf;
vector<uint8_t> Ag,Ab;
vector<uint8_t> Ee,A2;
static const int F0=2147483647;
static inline bool Dn(int f){return f<(int)Ee.size()&&Ee[f];}
vector<int> A4;
int A3;
int Cv;
double A7=0.85,A1=1.0;
int Ia;
double EL=3.0;
int Dj,Fa;
double At=0.85;
double Br,G0=1.0;
double Cb=1.0,AR;
double Fc,CV=1.0;
double BM,E6=1.0;
double Ef=0.25;
int Fb;
double Gt=0.5;
int BH;
double CB,BI=0.35;
double CU=0.96,Cs=0.035;
int Cu,Es=5;
double Fh=0.35,GQ=0.96;
double FW=0.035,Jk=0.15;
int Bj,DK;
double Ec;
int E4=1;
double H8=0.02;
int Dl;
double G7;
vector<array<float,3>> CQ;
double D3=1.0;
vector<int> D2;
int AL;
vector<double> Ae;
int AJ,IQ;
double AW;
int MA,G6=3;
double CO=0.10;
int BY,A5;
double IR=20.0;
const double IP=(0.01*255.0)*(0.01*255.0);
double Ba(double onx,double ony,double onz,
double nx,double ny,double nz){double Jl=0.0;
double a,b,d;
a=(onx+1.0)*127.5;b=(nx+1.0)*127.5;d=a-b;Jl+=d*d/(a*a+b*b+IP);
a=(ony+1.0)*127.5;b=(ny+1.0)*127.5;d=a-b;Jl+=d*d/(a*a+b*b+IP);
a=(onz+1.0)*127.5;b=(nz+1.0)*127.5;d=a-b;Jl+=d*d/(a*a+b*b+IP);
return Jl;}
double AQ(int GL,double cv,double FG,
double onx,double ony,double onz,
double nx,double ny,double nz){double nv=Ba(onx,ony,onz,nx,ny,nz);
if(GL!=2)return nv;
double wf=1.0/(1.0+IR*cv);
return(1.0-wf)*FG+wf*nv;}
double Ch,K1=1.0;
double Da;
vector<double> BP;
double DH=1.0,Dr;
vector<double> Av;
double JN;
vector<double> B7;
double Bq,If=0.15;
double Go=1.0;
vector<double> AA;
vector<double> Eu,Fm;
int H9;
int Cm,Bm=2;
double AN=0.35,Fp=1e-7;
int Ez;
double Hb=1.0,G2=1.0;
int A0,Cj,CS;
double Cw,B1,Cg,AB;
vector<array<float,3>> Ck;
vector<double> Bd;
int Gy,CW=4;
double EH=1e-4;
int CX;
int Be;
double AI=0.965;
double AO=0.035,Bu=45.0;
int Bk=1,Bl;static int Au;static int Co;static double A9=0.0495;
double BS;
int CK=-1,Gm;
int Fw=1,Dz=1024;
const double HM=800.0;
const double F3=2.5;
int DX;
float AC(float v){return DX?floorf(v+0.5f):v;}
struct Dd{vector<float> nr,ng,nb,dp;vector<uint8_t> cov;};
static Dd Bg[6];
const int Aw[6][2]={{0,1},{0,-1},{1,1},{1,-1},{2,1},{2,-1}};
vector<float> E5,I1,I2,I3,I4;
vector<uint8_t> HB;
vector<double> E8,FD,DD,DB,DC;
vector<uint8_t> Bt;
void AP(const vector<double>&Xv,const vector<double>&Yv,const vector<double>&Zv,
const vector<int>&fa,const vector<int>&fb,const vector<int>&fc,
const vector<uint8_t>&Hc,int nf,
int C7,int FL,int Al,
vector<float>&J5,vector<float>&J6,vector<float>&J7,
vector<float>&J8,vector<uint8_t>&Io){size_t N=(size_t)Al*Al;
J5.assign(N,AC(127.5f));J6.assign(N,AC(127.5f));J7.assign(N,AC(127.5f));
J8.assign(N,255.0f);Io.assign(N,0);
E5.assign(N,3.4028235e38f);
const double cu=Al/2.0,cv=Al/2.0;
const double focal=HM*((double)Al/1024.0);
double C[3]={0,0,0};C[C7]=FL*F3;
double f[3]={-C[0],-C[1],-C[2]};
double fn=sqrt(f[0]*f[0]+f[1]*f[1]+f[2]*f[2]);f[0]/=fn;f[1]/=fn;f[2]/=fn;
double up[3]={0,1,0};if(fabs(f[1])> 0.9){up[0]=0;up[1]=0;up[2]=1;}
double r[3]={f[1]*up[2]-f[2]*up[1],f[2]*up[0]-f[0]*up[2],f[0]*up[1]-f[1]*up[0]};
double rn=sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);r[0]/=rn;r[1]/=rn;r[2]/=rn;
double u[3]={r[1]*f[2]-r[2]*f[1],r[2]*f[0]-r[0]*f[2],r[0]*f[1]-r[1]*f[0]};
for(int i=0;i <nf;++i){if(!Hc.empty()&&!Hc[i]&&!Dn(i))continue;
int ia=fa[i],ib=fb[i],ic=fc[i];
if(ia==ib||ib==ic||ia==ic)continue;
double ax=Xv[ia],ay=Yv[ia],az=Zv[ia];
double bx=Xv[ib],by=Yv[ib],bz=Zv[ib];
double cx=Xv[ic],cy=Yv[ic],cz=Zv[ic];
double IC=bx-ax,IB=by-ay,IA=bz-az;
double I9=cx-ax,I8=cy-ay,I7=cz-az;
double Js=IB*I7-IA*I8,Jr=IA*I9-IC*I7,Jq=IC*I8-IB*I9;
double gl=sqrt(Js*Js+Jr*Jr+Jq*Jq);
if(gl <1e-30)continue;
Js/=gl;Jr/=gl;Jq/=gl;
float Rn=AC((float)((Js+1.0)*127.5));
float Gn=AC((float)((Jr+1.0)*127.5));
float Bn=AC((float)((Jq+1.0)*127.5));
double sx[3],sy[3],DO[3];bool ok=true;int GS[3]={ia,ib,ic};
for(int k=0;k<3;++k){double dx=Xv[GS[k]]-C[0],dy=Yv[GS[k]]-C[1],dz=Zv[GS[k]]-C[2];
double xc=dx*r[0]+dy*r[1]+dz*r[2];
double yc=dx*u[0]+dy*u[1]+dz*u[2];
double zc=dx*f[0]+dy*f[1]+dz*f[2];
if(zc <1e-9){ok=false;break;}
sx[k]=focal*xc/zc+cu;sy[k]=focal*yc/zc+cv;DO[k]=zc;}
if(!ok)continue;
double GN=min({sx[0],sx[1],sx[2]}),GP=max({sx[0],sx[1],sx[2]});
double GM=min({sy[0],sy[1],sy[2]}),GO=max({sy[0],sy[1],sy[2]});
int x0=max(0,(int)floor(GN)),x1=min(Al-1,(int)ceil(GP));
int y0=max(0,(int)floor(GM)),y1=min(Al-1,(int)ceil(GO));
if(x0>x1||y0>y1)continue;
double De=(sx[1]-sx[0])*(sy[2]-sy[0])-(sx[2]-sx[0])*(sy[1]-sy[0]);
if(fabs(De)<1e-12)continue;
double F6=1.0/De;
double Jp=1.0/DO[0],Jo=1.0/DO[1],Jn=1.0/DO[2];
for(int y=y0;y<=y1;++y){double py=y+0.5;
for(int x=x0;x<=x1;++x){double px=x+0.5;
double w0=((sx[1]-px)*(sy[2]-py)-(sx[2]-px)*(sy[1]-py))*F6;
double w1=((sx[2]-px)*(sy[0]-py)-(sx[0]-px)*(sy[2]-py))*F6;
double w2=1.0-w0-w1;
if(w0<0||w1<0||w2<0)continue;
double zp=1.0/(w0*Jp+w1*Jo+w2*Jn);
size_t Il=(size_t)y*Al+x;
if(zp<E5[Il]){E5[Il]=(float)zp;J8[Il]=AC((float)zp);
J5[Il]=Rn;J6[Il]=Gn;J7[Il]=Bn;Io[Il]=1;}}
}}
}
void Bb(vector<double>&P,int H,int W){size_t FM=(size_t)(W+1);
fill(P.begin(),P.begin()+FM,0.0);
for(int y=1;y <=H;++y)P[(size_t)y*FM]=0.0;}
double Ao(const vector<float>&A,const vector<float>&B,
const vector<uint8_t>&fg,int H,int W,
vector<double>&PA,vector<double>&PB,
vector<double>&HZ,vector<double>&KE,vector<double>&KF){const double C1=(0.01*255.0)*(0.01*255.0);
const double C2=(0.03*255.0)*(0.03*255.0);
const int M9=11,He=M9/2;
size_t FM=(size_t)(W+1);
Bb(PA,H,W);Bb(PB,H,W);
Bb(HZ,H,W);Bb(KE,H,W);
Bb(KF,H,W);
for(int y=1;y<=H;++y){size_t rp=(size_t)y*FM,IS=(size_t)(y-1)*FM,ra=(size_t)(y-1)*W;
for(int x=1;x<=W;++x){double a=A[ra+(x-1)],b=B[ra+(x-1)];
PA[rp+x]=a+PA[IS+x]+PA[rp+x-1]-PA[IS+x-1];
PB[rp+x]=b+PB[IS+x]+PB[rp+x-1]-PB[IS+x-1];
HZ[rp+x]=a*a+HZ[IS+x]+HZ[rp+x-1]-HZ[IS+x-1];
KE[rp+x]=b*b+KE[IS+x]+KE[rp+x-1]-KE[IS+x-1];
KF[rp+x]=a*b+KF[IS+x]+KF[rp+x-1]-KF[IS+x-1];}}
auto II=[&](const vector<double>&P,int r0,int c0,int r1,int c1)->double{return P[(size_t)(r1+1)*FM+(c1+1)]-P[(size_t)r0*FM+(c1+1)]
-P[(size_t)(r1+1)*FM+c0]+P[(size_t)r0*FM+c0];};
const double n=M9*M9;double Lx=0;long Eh=0;
for(int cy=He;cy<H-He;++cy)
for(int cx=He;cx<W-He;++cx){if(!fg[(size_t)cy*W+cx])continue;
int r0=cy-He,c0=cx-He,r1=cy+He,c1=cx+He;
double K8=II(PA,r0,c0,r1,c1)/n,K7=II(PB,r0,c0,r1,c1)/n;
double vA=II(HZ,r0,c0,r1,c1)/n-K8*K8;
double vB=II(KE,r0,c0,r1,c1)/n-K7*K7;
double Nu=II(KF,r0,c0,r1,c1)/n-K8*K7;
double s=((2*K8*K7+C1)*(2*Nu+C2))/((K8*K8+K7*K7+C1)*(vA+vB+C2));
Lx+=s;++Eh;}
return Eh?Lx/Eh:1.0;}
int Ij,GH=-1;
void EA(){Ij=Dz;GH=DX;
vector<uint8_t> Jg;
for(int vi=0;vi<6;++vi){AP(oX,oY,oZ,L3,L2,L1,Jg,NF,
Aw[vi][0],Aw[vi][1],Dz,
Bg[vi].nr,Bg[vi].ng,Bg[vi].nb,
Bg[vi].dp,Bg[vi].cov);}}
double Hu(){int H=Dz,W=Dz;
size_t IN=(size_t)(H+1)*(W+1),N=(size_t)H*W;
if(E8.size()!=IN){E8.resize(IN);FD.resize(IN);DD.resize(IN);DB.resize(IN);DC.resize(IN);}
if(Bt.size()!=N)Bt.resize(N);
double I5=0.0;
for(int vi=0;vi<6;++vi){AP(X,Y,Z,FA,FB,FC,Ab,NF,
Aw[vi][0],Aw[vi][1],Dz,
I1,I2,I3,I4,HB);
const Dd&R=Bg[vi];
for(size_t i=0;i<N;++i)Bt[i]=(R.cov[i]||HB[i])?1:0;
double sr=Ao(R.nr,I1,Bt,H,W,E8,FD,DD,DB,DC);
double sg=Ao(R.ng,I2,Bt,H,W,E8,FD,DD,DB,DC);
double sb=Ao(R.nb,I3,Bt,H,W,E8,FD,DD,DB,DC);
double sd=Ao(R.dp,I4,Bt,H,W,E8,FD,DD,DB,DC);
double vN=(sr+sg+sb)/3.0;
I5+=0.5*vN+0.5*sd;}
return I5/6.0;}
vector<int> uf;
int H2(int x){while(uf[x]!=x){uf[x]=uf[uf[x]];x=uf[x];}return x;}
void Ak(array<double,10>&a,const array<double,10>&b){for(int i=0;i <10;++i)a[i]+=b[i];}
array<double,10> B6(int a,int b,int c){double IC=X[b]-X[a],IB=Y[b]-Y[a],IA=Z[b]-Z[a];
double I9=X[c]-X[a],I8=Y[c]-Y[a],I7=Z[c]-Z[a];
double nx=IB*I7-IA*I8;
double ny=IA*I9-IC*I7;
double nz=IC*I8-IB*I9;
double Lb=sqrt(nx*nx+ny*ny+nz*nz);
array<double,10> q{};q.fill(0.0);
if(Lb <1e-30)return q;
double De=0.5*Lb;
nx/=Lb;ny/=Lb;nz/=Lb;
double C8;
double Gb=max({abs(nx),abs(ny),abs(nz)});
if(Ia==1){C8=1.0+EL*(abs(nx)+abs(ny)+abs(nz)-1.0);}else if(Ia==2){C8=1.0+EL*Gb*Gb+JN*(1.0-Gb*Gb);}else{
C8=1.0+EL*Gb*Gb;}
double d=-(nx*X[a]+ny*Y[a]+nz*Z[a]);
double w=De*C8;
double A=nx,B=ny,Cc=nz,D=d;
q[0]=w*A*A;q[1]=w*A*B;q[2]=w*A*Cc;q[3]=w*A*D;
q[4]=w*B*B;q[5]=w*B*Cc;q[6]=w*B*D;
q[7]=w*Cc*Cc;q[8]=w*Cc*D;
q[9]=w*D*D;
return q;}
void Ai(int v){array<double,10> q;q.fill(0.0);
for(int f:vf[v]){if(!Ab[f])continue;
int a=FA[f],b=FB[f],c=FC[f];
if(a!=v&&b!=v&&c!=v)continue;
if(a==b||b==c||a==c)continue;
array<double,10> fq=B6(a,b,c);
double m=(f <(int)Bd.size())?Bd[f]:1.0;
for(int k=0;k<10;++k)q[k]+=fq[k]*m;}
Q[v]=q;}
double AG(const array<double,10>&q,double x,double y,double z){return q[0]*x*x+2*q[1]*x*y+2*q[2]*x*z+2*q[3]*x
+q[4]*y*y+2*q[5]*y*z+2*q[6]*y
+q[7]*z*z+2*q[8]*z+q[9];}
bool Eq(const array<double,10>&q,double&ox,double&oy,double&oz){double LK=q[0]+1e-9,JA=q[1],J9=q[2];
double KZ=q[4]+1e-9,Jj=q[5],KY=q[7]+1e-9;
double b0=-q[3],b1=-q[6],b2=-q[8];
double MT=LK*(KZ*KY-Jj*Jj)
-JA*(JA*KY-Jj*J9)
+J9*(JA*Jj-KZ*J9);
if(fabs(MT)<1e-12)return false;
double F6=1.0/MT;
double NL=(KZ*KY-Jj*Jj);
double Ng=-(JA*KY-Jj*J9);
double Nf=(JA*Jj-KZ*J9);
double NK=(LK*KY-J9*J9);
double Nb=-(LK*Jj-JA*J9);
double NJ=(LK*KZ-JA*JA);
ox=(NL*b0+Ng*b1+Nf*b2)*F6;
oy=(Ng*b0+NK*b1+Nb*b2)*F6;
oz=(Nf*b0+Nb*b1+NJ*b2)*F6;
return true;}
void Am(int u,int v,const array<double,10>&q,
double&px,double&py,double&pz,double&cost){double ox,oy,oz;
if(Eq(q,ox,oy,oz)){px=ox;py=oy;pz=oz;
cost=AG(q,ox,oy,oz);
if(cost <0)cost=0;
return;}
double e0=AG(q,X[u],Y[u],Z[u]),e1=AG(q,X[v],Y[v],Z[v]);
double e2=AG(q,.5*(X[u]+X[v]),.5*(Y[u]+Y[v]),.5*(Z[u]+Z[v]));
cost=e0;px=X[u];py=Y[u];pz=Z[u];
if(e1<cost){cost=e1;px=X[v];py=Y[v];pz=Z[v];}
if(e2<cost){cost=e2;px=.5*(X[u]+X[v]);py=.5*(Y[u]+Y[v]);pz=.5*(Z[u]+Z[v]);}
if(cost <0)cost=0;}
vector<int> Cp;
int BF;
vector<int> B5;
int CM=1;
void As(int u,vector<int>&Ip){Ip.clear();
++BF;
for(int f:vf[u]){if(!Ab[f])continue;
int a=FA[f],b=FB[f],c=FC[f];
if(a!=u&&b!=u&&c!=u)continue;
if(a!=u&&Cp[a]!=BF){Cp[a]=BF;Ip.push_back(a);}
if(b!=u&&Cp[b]!=BF){Cp[b]=BF;Ip.push_back(b);}
if(c!=u&&Cp[c]!=BF){Cp[c]=BF;Ip.push_back(c);}}
}
int Ar(int u,int v,int fs[2],int Gu[2]){int Eh=0;
const vector<int>&GA=(vf[u].size()<=vf[v].size())?vf[u]:vf[v];
for(int f:GA){if(!Ab[f])continue;
int a=FA[f],b=FB[f],c=FC[f];
bool Mi=(a==u||b==u||c==u);
bool Mh=(a==v||b==v||c==v);
if(Mi&&Mh){if(Eh <2){fs[Eh]=f;
Gu[Eh]=(a!=u&&a!=v)?a:((b!=u&&b!=v)?b:c);}
++Eh;}}
return Eh;}
bool AZ(int u,int v,const int Gu[2]){if((int)B5.size()!=NV)B5.assign(NV,0);
if(CM> 2000000000){fill(B5.begin(),B5.end(),0);
CM=1;}
int Ji=CM+1;
int Jw=CM+2;
CM+=2;
for(int f:vf[v]){if(!Ab[f])continue;
int a=FA[f],b=FB[f],c=FC[f];
if(a!=v&&b!=v&&c!=v)continue;
if(a!=v)B5[a]=Ji;
if(b!=v)B5[b]=Ji;
if(c!=v)B5[c]=Ji;}
int Fr=0;
for(int f:vf[u]){if(!Ab[f])continue;
int a=FA[f],b=FB[f],c=FC[f];
if(a!=u&&b!=u&&c!=u)continue;
int xs[3]={a,b,c};
for(int i=0;i<3;++i){int x=xs[i];
if(x==u||B5[x]!=Ji)continue;
B5[x]=Jw;
++Fr;
if(x!=Gu[0]&&x!=Gu[1])return false;}}
return Fr==2;}
bool AE(int f,int v){return FA[f]==v||FB[f]==v||FC[f]==v;}
int DG(int f,int x,int y){const vector<int>&GA=(vf[x].size()<=vf[y].size())?vf[x]:vf[y];
for(int g:GA){if(g==f)continue;
if(!Ab.empty()&&!Ab[g])continue;
if(AE(g,x)&&AE(g,y))return g;}
return-1;}
struct AH{
double cost;
int u,v,ver_u,ver_v;
bool operator<(const AH&o)const{return cost> o.cost;}};
const double Gc=1e-9;
bool Af(double ax,double ay,double az,double bx,double by,double bz,
double cx,double cy,double cz,
double&nx,double&ny,double&nz){double IC=bx-ax,IB=by-ay,IA=bz-az;
double I9=cx-ax,I8=cy-ay,I7=cz-az;
nx=IB*I7-IA*I8;ny=IA*I9-IC*I7;nz=IC*I8-IB*I9;
double l=sqrt(nx*nx+ny*ny+nz*nz);
if(l <Gc)return false;
nx/=l;ny/=l;nz/=l;return true;}
void DS(){Ck.assign(NF,{0.f,0.f,0.f});
for(int f=0;f <NF;++f){int a=L3[f],b=L2[f],c=L1[f];
double nx,ny,nz;
if(Af(oX[a],oY[a],oZ[a],oX[b],oY[b],oZ[b],
oX[c],oY[c],oZ[c],nx,ny,nz))
Ck[f]={(float)nx,(float)ny,(float)nz};}}
bool BG(int f,double&nx,double&ny,double&nz){if(f <0||f>=(int)Ck.size())return false;
nx=Ck[f][0];ny=Ck[f][1];nz=Ck[f][2];
return nx*nx+ny*ny+nz*nz> 0.5;}
static inline double CA(double px,double py,double pz,
double ax,double ay,double az,double bx,double by,double bz,double cx,double cy,double cz){
double KD=bx-ax,KC=by-ay,KB=bz-az;
double Lw=cx-ax,Lv=cy-ay,Lu=cz-az;
double Nk=px-ax,Nj=py-ay,Ni=pz-az;
double d1=Nk*KD+Nj*KC+Ni*KB;
double d2=Nk*Lw+Nj*Lv+Ni*Lu;
double aa=KD*KD+KC*KC+KB*KB;
double bb=Lw*Lw+Lv*Lv+Lu*Lu;
double Jz=KD*Lw+KC*Lv+KB*Lu;
double Mw=aa*bb-Jz*Jz;
double qx,qy,qz,dx,dy,dz;
if(Mw<1e-30){double t=d1/(aa>1e-30?aa:1e-30); if(t<0)t=0; if(t>1)t=1;
qx=ax+t*KD;qy=ay+t*KC;qz=az+t*KB;
dx=px-qx;dy=py-qy;dz=pz-qz;return dx*dx+dy*dy+dz*dz;}
double v=(bb*d1-Jz*d2)/Mw, w=(aa*d2-Jz*d1)/Mw;
if(v<0)v=0;
if(v>1)v=1;
if(w<0)w=0;
if(w>1)w=1;
double s=v+w; if(s>1){v/=s;w/=s;}
qx=ax+v*KD+w*Lw;qy=ay+v*KC+w*Lv;qz=az+v*KB+w*Lu;
dx=px-qx;dy=py-qy;dz=pz-qz;return dx*dx+dy*dy+dz*dz;}
struct J0{
double Ei=1.0,ox=0,oy=0,oz=0; int nx=1,ny=1,nz=1; long nc=1;
vector<int> Ie; vector<int> Iz; bool KP=false;
const double *VX=nullptr,*VY=nullptr,*VZ=nullptr;
const int *Ns=nullptr,*Nr=nullptr,*Nq=nullptr;
inline long Ey(int ix,int iy,int iz)const{return ((long)iz*ny+iy)*nx+ix;}
inline void Ex(double x,double y,double z,int&ix,int&iy,int&iz)const{
ix=(int)((x-ox)/Ei); iy=(int)((y-oy)/Ei); iz=(int)((z-oz)/Ei);
if(ix<0)ix=0;
if(ix>=nx)ix=nx-1;
if(iy<0)iy=0;
if(iy>=ny)iy=ny-1;
if(iz<0)iz=0;
if(iz>=nz)iz=nz-1;}
void buildFrom(const double*vx,const double*vy,const double*vz,int nv,
const int*fa,const int*fb,const int*fc,const uint8_t*Hc,int nf,double ID){
VX=vx;VY=vy;VZ=vz;Ns=fa;Nr=fb;Nq=fc;
double Fg=1e300,Ff=1e300,Hp=1e300,F5=-1e300,F4=-1e300,H7=-1e300;
for(int i=0;i<nv;++i){double x=vx[i],y=vy[i],z=vz[i];
Fg=min(Fg,x);F5=max(F5,x);Ff=min(Ff,y);F4=max(F4,y);Hp=min(Hp,z);H7=max(H7,z);}
double ex=F5-Fg,ey=F4-Ff,ez=H7-Hp;
Ei=ID/40.0; if(Ei<=0)Ei=1.0;
ox=Fg;oy=Ff;oz=Hp;
nx=(int)(ex/Ei)+1; ny=(int)(ey/Ei)+1; nz=(int)(ez/Ei)+1;
while((double)nx*ny*nz>4.0e6){Ei*=1.5; nx=(int)(ex/Ei)+1; ny=(int)(ey/Ei)+1; nz=(int)(ez/Ei)+1;}
nc=(long)nx*ny*nz;
auto Ci=[&](int f)->bool{if(Hc&&!Hc[f])return false;int a=fa[f],b=fb[f],c=fc[f];return !(a==b||b==c||a==c);};
vector<int> Eh(nc+1,0);
for(int f=0;f<nf;++f){if(!Ci(f))continue;int a=fa[f],b=fb[f],c=fc[f];
double gx=(vx[a]+vx[b]+vx[c])/3.0,gy=(vy[a]+vy[b]+vy[c])/3.0,gz=(vz[a]+vz[b]+vz[c])/3.0;
int ix,iy,iz; Ex(gx,gy,gz,ix,iy,iz); ++Eh[Ey(ix,iy,iz)+1];}
for(long i=0;i<nc;++i)Eh[i+1]+=Eh[i];
Ie=Eh; Iz.assign(Eh[nc],0);
vector<int> Df(Eh.begin(),Eh.end());
for(int f=0;f<nf;++f){if(!Ci(f))continue;int a=fa[f],b=fb[f],c=fc[f];
double gx=(vx[a]+vx[b]+vx[c])/3.0,gy=(vy[a]+vy[b]+vy[c])/3.0,gz=(vz[a]+vz[b]+vz[c])/3.0;
int ix,iy,iz; Ex(gx,gy,gz,ix,iy,iz); long ci=Ey(ix,iy,iz); Iz[Df[ci]++]=f;}
KP=true;}
double nearest2(double px,double py,double pz,double HT)const{(void)HT;
int cx,cy,cz; Ex(px,py,pz,cx,cy,cz);
double AU=1e300;
int Mj=nx+ny+nz+2;
for(int r=0;;++r){
int x0=cx-r,x1=cx+r,y0=cy-r,y1=cy+r,z0=cz-r,z1=cz+r;
for(int iz=z0;iz<=z1;++iz){if(iz<0||iz>=nz)continue;
for(int iy=y0;iy<=y1;++iy){if(iy<0||iy>=ny)continue;
for(int ix=x0;ix<=x1;++ix){if(ix<0||ix>=nx)continue;
if(r>0&&ix>x0&&ix<x1&&iy>y0&&iy<y1&&iz>z0&&iz<z1)continue;
long ci=Ey(ix,iy,iz);
for(int k=Ie[ci];k<Ie[ci+1];++k){int f=Iz[k];
int a=Ns[f],b=Nr[f],c=Nq[f];
double d2=CA(px,py,pz,VX[a],VY[a],VZ[a],VX[b],VY[b],VZ[b],VX[c],VY[c],VZ[c]);
if(d2<AU)AU=d2;}}}}
if(r>=1){
if(AU<1e299){double m=(r-0.99)*Ei; if(AU<=m*m||r>6)break;}
if(r>8)break;}
if(r>=Mj)break;
}
return AU;}
};
double AS(int f,int GL,
double onx,double ony,double onz,
double nx,double ny,double nz){double Jd=max(-1.0,min(1.0,onx*nx+ony*ny+onz*nz));
double DU=1.0-Jd;
if(GL)
DU=AQ(GL,(f<(int)Av.size()?Av[f]:0.0),
DU,onx,ony,onz,nx,ny,nz);
if(B1 <=0.0)return DU;
double FP,FO,FN;
if(!BG(f,FP,FO,FN))return DU;
double JZ=max(-1.0,min(1.0,FP*nx+FO*ny+FN*nz));
double ref=1.0-JZ;
if(GL)
ref=AQ(GL,(f<(int)Av.size()?Av[f]:0.0),
ref,FP,FO,FN,nx,ny,nz);
double b=max(0.0,min(1.0,B1));
return(1.0-b)*DU+b*ref;}
bool BJ(int u,int v,int f0,int f1,double px,double py,double pz){for(int A6=0;A6 <2;++A6){int CP=A6==0?u:v;
for(int f:vf[CP]){if(!Ab[f]||f==f0||f==f1)continue;
int a=FA[f],b=FB[f],c=FC[f];
if(A6==1&&(a==u||b==u||c==u))continue;
double onx,ony,onz;
if(!Af(X[a],Y[a],Z[a],X[b],Y[b],Z[b],X[c],Y[c],Z[c],onx,ony,onz))
continue;
auto gx=[&](int w){return(w==v||w==u)?px:X[w];};
auto gy=[&](int w){return(w==v||w==u)?py:Y[w];};
auto gz=[&](int w){return(w==v||w==u)?pz:Z[w];};
int cu=(a==u)+(b==u)+(c==u);
int cv=(a==v)+(b==v)+(c==v);
if(cu&&cv)return false;
double nx,ny,nz;
if(!Af(gx(a),gy(a),gz(a),gx(b),gy(b),gz(b),gx(c),gy(c),gz(c),nx,ny,nz))
return false;
if(onx*nx+ony*ny+onz*nz <At)return false;}}
return true;}
int EN=768;
double BU=600.0,EO=2.5;
int Cn,Ed,Ct;
vector<float> Gg;
vector<int> Ib,CI;
void BZ(){const int R=EN;
const double MX=R/2.0;
double Fg=1e300,Ff=1e300,Hp=1e300,F5=-1e300,F4=-1e300,H7=-1e300;
for(int i=0;i<NV;++i){Fg=min(Fg,X[i]);F5=max(F5,X[i]);Ff=min(Ff,Y[i]);F4=max(F4,Y[i]);Hp=min(Hp,Z[i]);H7=max(H7,Z[i]);}
double IV[3]={(Fg+F5)*0.5,(Ff+F4)*0.5,(Hp+H7)*0.5};
if(Cn){IV[0]=IV[1]=IV[2]=0.0;}
double LA=0.5*sqrt((F5-Fg)*(F5-Fg)+(F4-Ff)*(F4-Ff)+(H7-Hp)*(H7-Hp));
if(LA<1e-30)LA=1.0;
const double Mv=(Cn?EO:EO*LA),focal=BU;
D2.assign(NF,0);
Ae.assign(NF,0.0);
Eu.assign(NF,0.0);
Fm.assign(NF,0.0);
B7.assign(NF,0.0);
const double Hw=If*LA;
const size_t RN=(size_t)R*R;
if(Gg.size()!=RN){Gg.resize(RN);Ib.resize(RN);}
if((int)CI.size()!=NF)CI.assign(NF,0);else fill(CI.begin(),CI.end(),0);
auto&HN=Gg;auto&fid=Ib;auto&DI=CI;
for(int C7=0;C7 <3;++C7)for(int FL=-1;FL <=1;FL+=2){double C[3]={IV[0],IV[1],IV[2]};C[C7]=IV[C7]+FL*Mv;
double f[3]={IV[0]-C[0],IV[1]-C[1],IV[2]-C[2]};double fn=sqrt(f[0]*f[0]+f[1]*f[1]+f[2]*f[2]);f[0]/=fn;f[1]/=fn;f[2]/=fn;
double up[3]={0,1,0};if(fabs(f[1])>0.9){up[0]=0;up[1]=0;up[2]=1;}
double r[3]={f[1]*up[2]-f[2]*up[1],f[2]*up[0]-f[0]*up[2],f[0]*up[1]-f[1]*up[0]};
double rn=sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);r[0]/=rn;r[1]/=rn;r[2]/=rn;
double u[3]={r[1]*f[2]-r[2]*f[1],r[2]*f[0]-r[0]*f[2],r[0]*f[1]-r[1]*f[0]};
fill(HN.begin(),HN.end(),1e30f);
fill(fid.begin(),fid.end(),-1);
for(int i=0;i<NF;++i){if(!Ab.empty()&&!Ab[i])continue;
int GS[3]={FA[i],FB[i],FC[i]};double sx[3],sy[3],DO[3];bool ok=true;
for(int k=0;k<3;++k){int vi=GS[k];double dx=X[vi]-C[0],dy=Y[vi]-C[1],dz=Z[vi]-C[2];
double xc=dx*r[0]+dy*r[1]+dz*r[2],yc=dx*u[0]+dy*u[1]+dz*u[2],d=dx*f[0]+dy*f[1]+dz*f[2];
if(d<1e-6){ok=false;break;}sx[k]=focal*xc/d+MX;sy[k]=focal*yc/d+MX;DO[k]=d;}
if(!ok)continue;
double GN=min({sx[0],sx[1],sx[2]}),GP=max({sx[0],sx[1],sx[2]});
double GM=min({sy[0],sy[1],sy[2]}),GO=max({sy[0],sy[1],sy[2]});
int x0=max(0,(int)floor(GN)),x1=min(R-1,(int)ceil(GP));
int y0=max(0,(int)floor(GM)),y1=min(R-1,(int)ceil(GO));
if(x0>x1||y0>y1)continue;
double De=(sx[1]-sx[0])*(sy[2]-sy[0])-(sx[2]-sx[0])*(sy[1]-sy[0]);
if(fabs(De)<1e-12)continue;
double F6=1.0/De;
double Jp=1/DO[0],Jo=1/DO[1],Jn=1/DO[2];
for(int y=y0;y<=y1;++y){double py=y+0.5;for(int x=x0;x<=x1;++x){double px=x+0.5;
double w0=((sx[1]-px)*(sy[2]-py)-(sx[2]-px)*(sy[1]-py))*F6;
double w1=((sx[2]-px)*(sy[0]-py)-(sx[0]-px)*(sy[2]-py))*F6;
double w2=1-w0-w1;if(w0<0||w1<0||w2<0)continue;
double zp=1.0/(w0*Jp+w1*Jo+w2*Jn);size_t Il=(size_t)y*R+x;
if(zp<HN[Il]){HN[Il]=(float)zp;fid[Il]=i;}}}}
for(size_t p=0;p<(size_t)R*R;++p)if(fid[p]>=0){DI[fid[p]]++;}
if(Bq> 0.0){for(int y=0;y <R;++y)for(int x=0;x <R;++x){size_t p=(size_t)y*R+x;int ff=fid[p];
if(ff <0)continue;
float zf=HN[p];bool Eg=false;
const int NG[4]={-1,1,0,0},NE[4]={0,0,-1,1};
for(int k=0;k <4&&!Eg;++k){int Mc=x+NG[k],Mb=y+NE[k];
if(Mc<0||Mc>=R||Mb<0||Mb>=R){Eg=true;break;}
size_t q=(size_t)Mb*R+Mc;int nf=fid[q];
if(nf <0){Eg=true;break;}
if(nf!=ff&&fabs((double)HN[q]-(double)zf)> Hw)Eg=true;}
if(Eg)B7[ff]+=1.0;}}
double Gh=0.0;
for(int f=0;f<NF;++f)Gh+=DI[f];
double KH=(AL&&Gh> 0.0)?(1.0/Gh):1.0;
for(int f=0;f<NF;++f)if(DI[f]>0){D2[f]+=DI[f];Ae[f]+=(double)DI[f]*KH;
Eu[f]+=1.0;Fm[f]+=sqrt((double)DI[f]);
DI[f]=0;}}
}
void En(){if(Bq <=0.0||(int)B7.size()!=NF)return;
double LO=0.0;long Eh=0;
for(int f=0;f<NF;++f)if(Ab.empty()||Ab[f]){LO+=B7[f];if(B7[f]>0.0)++Eh;}
Go=(Eh>0&&LO>0.0)?LO/Eh:1.0;
AA.assign(NV,0.0);
for(int f=0;f<NF;++f)if((Ab.empty()||Ab[f])&&B7[f]>0.0){double sv=B7[f]/Go;
AA[FA[f]]=max(AA[FA[f]],sv);
AA[FB[f]]=max(AA[FB[f]],sv);
AA[FC[f]]=max(AA[FC[f]],sv);}}
double Bh(int u,int v,int f0,int f1,double px,double py,double pz){double A8=0.0;
for(int A6=0;A6 <2;++A6){int CP=A6==0?u:v;
for(int f:vf[CP]){if(!Ab[f]||f==f0||f==f1)continue;
int a=FA[f],b=FB[f],c=FC[f];
if(A6==1&&(a==u||b==u||c==u))continue;
double onx,ony,onz;
if(!Af(X[a],Y[a],Z[a],X[b],Y[b],Z[b],X[c],Y[c],Z[c],onx,ony,onz))
continue;
auto gx=[&](int w){return(w==v||w==u)?px:X[w];};
auto gy=[&](int w){return(w==v||w==u)?py:Y[w];};
auto gz=[&](int w){return(w==v||w==u)?pz:Z[w];};
double nx,ny,nz;
if(!Af(gx(a),gy(a),gz(a),gx(b),gy(b),gz(b),gx(c),gy(c),gz(c),nx,ny,nz))
{A8+=2.0;continue;}
A8+=1.0-(onx*nx+ony*ny+onz*nz);}}
return A8;}
int B2;
double Ej(double ax,double ay,double az){return fabs(ax)+fabs(ay)+fabs(az);}
double AY(int u,int v,int f0,int f1,
double px,double py,double pz,
bool Db,bool F9,
int GK){if((Db?AB:Cg)<=0.0)
return 0.0;
auto gx=[&](int w){return(w==u||w==v)?px:X[w];};
auto gy=[&](int w){return(w==u||w==v)?py:Y[w];};
auto gz=[&](int w){return(w==u||w==v)?pz:Z[w];};
int AK[2]={f0,f1};
double A8=0.0;
for(int ri=0;ri<2;++ri){int rf=AK[ri];
double FP,FO,FN;
if(!BG(rf,FP,FO,FN)){int a=FA[rf],b=FB[rf],c=FC[rf];
if(!Af(X[a],Y[a],Z[a],X[b],Y[b],Z[b],X[c],Y[c],Z[c],
FP,FO,FN))
continue;}
double AU=2.0;
bool JB=false;
for(int A6=0;A6<2;++A6){int CP=(A6==0?u:v);
for(int f:vf[CP]){if(!Ab[f]||f==f0||f==f1)continue;
int a=FA[f],b=FB[f],c=FC[f];
if(A6==1&&(a==u||b==u||c==u))continue;
double nx,ny,nz;
if(!Af(gx(a),gy(a),gz(a),gx(b),gy(b),gz(b),
gx(c),gy(c),gz(c),nx,ny,nz))
continue;
double Mu=max(-1.0,min(1.0,FP*nx+FO*ny+FN*nz));
double d=1.0-Mu;
if(GK)
d=AQ(GK,
(rf<(int)Av.size()?Av[rf]:0.0),
d,FP,FO,FN,nx,ny,nz);
if(d<AU)AU=d;
JB=true;}}
if(!JB)AU=2.0;
if(F9&&Fb==1)
AU=pow(2.0*max(0.0,AU)+1e-12,Gt);
double w=0.0;
if(Db){if(AJ){w=(rf<(int)Ae.size())?Ae[rf]:0.0;}else{
int a=FA[rf],b=FB[rf],c=FC[rf];
double ax=(Y[b]-Y[a])*(Z[c]-Z[a])-(Z[b]-Z[a])*(Y[c]-Y[a]);
double ay=(Z[b]-Z[a])*(X[c]-X[a])-(X[b]-X[a])*(Z[c]-Z[a]);
double az=(X[b]-X[a])*(Y[c]-Y[a])-(Y[b]-Y[a])*(X[c]-X[a]);
w=Ej(ax,ay,az);}}else{
double cov=(rf<(int)Ae.size())?Ae[rf]:0.0;
w=pow((cov/D3)+1e-6,G0);
if(AR>0.0&&rf<(int)Av.size())
w/=(1.0+AR*Av[rf]);}
A8+=w*AU;}
return A8;}
double BE(int u,int v,int f0,int f1,
double px,double py,double pz){double A8=0.0;
for(int A6=0;A6 <2;++A6){int CP=A6==0?u:v;
for(int f:vf[CP]){if(!Ab[f]||f==f0||f==f1)continue;
int a=FA[f],b=FB[f],c=FC[f];
if(A6==1&&(a==u||b==u||c==u))continue;
double Ko=(Y[b]-Y[a])*(Z[c]-Z[a])-(Z[b]-Z[a])*(Y[c]-Y[a]);
double Kn=(Z[b]-Z[a])*(X[c]-X[a])-(X[b]-X[a])*(Z[c]-Z[a]);
double Km=(X[b]-X[a])*(Y[c]-Y[a])-(Y[b]-Y[a])*(X[c]-X[a]);
double Gv=sqrt(Ko*Ko+Kn*Kn+Km*Km);
if(Gv <Gc)continue;
double onx=Ko/Gv,ony=Kn/Gv,onz=Km/Gv;
auto gx=[&](int w){return(w==v||w==u)?px:X[w];};
auto gy=[&](int w){return(w==v||w==u)?py:Y[w];};
auto gz=[&](int w){return(w==v||w==u)?pz:Z[w];};
double Ax=gx(a),Ay=gy(a),Az=gz(a);
double Bx=gx(b),By=gy(b),Bz=gz(b);
double Cx=gx(c),Cy=gy(c),Cz=gz(c);
double FU=(By-Ay)*(Cz-Az)-(Bz-Az)*(Cy-Ay);
double FT=(Bz-Az)*(Cx-Ax)-(Bx-Ax)*(Cz-Az);
double FS=(Bx-Ax)*(Cy-Ay)-(By-Ay)*(Cx-Ax);
double Dk=sqrt(FU*FU+FT*FT+FS*FS);
double w=AJ?((f <(int)Ae.size())?Ae[f]:0.0)
:Ej(FU,FT,FS);
double C0;
if(Dk <Gc)C0=2.0;
else C0=AS(f,A5,onx,ony,onz,
FU/Dk,FT/Dk,FS/Dk);
A8+=w*C0;}}
if(AB> 0.0)
A8+=AB*AY(
u,v,f0,f1,px,py,pz,true,false,A5);
return A8;}
struct Fy{
double onx,ony,onz,rpcW,nbpW;
int f,a,b,c;
uint8_t sa,sb,sc,ov;};
vector<Fy> Dw;
bool CR;
int GG[2];
double Gr[2][3],JY[2],H6[2];
void B8(int u,int v,int f0,int f1){Dw.clear();
CR=false;
for(int A6=0;A6 <2;++A6){int CP=A6==0?u:v;
for(int f:vf[CP]){if(!Ab[f]||f==f0||f==f1)continue;
int a=FA[f],b=FB[f],c=FC[f];
if(A6==1&&(a==u||b==u||c==u))continue;
Fy r;
r.f=f;r.a=a;r.b=b;r.c=c;
r.sa=(a==u||a==v);r.sb=(b==u||b==v);r.sc=(c==u||c==v);
double Ko=(Y[b]-Y[a])*(Z[c]-Z[a])-(Z[b]-Z[a])*(Y[c]-Y[a]);
double Kn=(Z[b]-Z[a])*(X[c]-X[a])-(X[b]-X[a])*(Z[c]-Z[a]);
double Km=(X[b]-X[a])*(Y[c]-Y[a])-(Y[b]-Y[a])*(X[c]-X[a]);
double Gv=sqrt(Ko*Ko+Kn*Kn+Km*Km);
r.ov=(Gv <Gc)?0:1;
if(r.ov){r.onx=Ko/Gv;r.ony=Kn/Gv;r.onz=Km/Gv;
int cu=(a==u)+(b==u)+(c==u),cv=(a==v)+(b==v)+(c==v);
if(cu&&cv)CR=true;}else{r.onx=r.ony=r.onz=0.0;}
r.rpcW=(f <(int)Ae.size())?Ae[f]:0.0;
r.nbpW=(f <(int)Ae.size()&&Ae[f]> 0.0)?Ae[f]:1.0;
Dw.push_back(r);}}
GG[0]=GG[1]=0;
if(AB> 0.0){int AK[2]={f0,f1};
for(int ri=0;ri<2;++ri){int rf=AK[ri];
double FP,FO,FN;
if(!BG(rf,FP,FO,FN)){int a=FA[rf],b=FB[rf],c=FC[rf];
if(!Af(X[a],Y[a],Z[a],X[b],Y[b],Z[b],X[c],Y[c],Z[c],
FP,FO,FN))
continue;}
GG[ri]=1;
Gr[ri][0]=FP;Gr[ri][1]=FO;Gr[ri][2]=FN;
if(AJ){JY[ri]=(rf<(int)Ae.size())?Ae[rf]:0.0;}else{
int a=FA[rf],b=FB[rf],c=FC[rf];
double ax=(Y[b]-Y[a])*(Z[c]-Z[a])-(Z[b]-Z[a])*(Y[c]-Y[a]);
double ay=(Z[b]-Z[a])*(X[c]-X[a])-(X[b]-X[a])*(Z[c]-Z[a]);
double az=(X[b]-X[a])*(Y[c]-Y[a])-(Y[b]-Y[a])*(X[c]-X[a]);
JY[ri]=Ej(ax,ay,az);}
H6[ri]=(rf<(int)Av.size()?Av[rf]:0.0);}}
}
bool BV(int u,int v,double px,double py,double pz,
double&KQ,double&Kp){if(CR)return false;
double Gq=0.0,Iw=0.0;int Ho=0;
double Ht[2]={2.0,2.0};
const bool JE=AB> 0.0;
const bool Lj=IQ!=0;
const int n=(int)Dw.size();
for(int i=0;i <n;++i){const Fy&r=Dw[i];
double Ax=r.sa?px:X[r.a],Ay=r.sa?py:Y[r.a],Az=r.sa?pz:Z[r.a];
double Bx=r.sb?px:X[r.b],By=r.sb?py:Y[r.b],Bz=r.sb?pz:Z[r.b];
double Cx=r.sc?px:X[r.c],Cy=r.sc?py:Y[r.c],Cz=r.sc?pz:Z[r.c];
double FU=(By-Ay)*(Cz-Az)-(Bz-Az)*(Cy-Ay);
double FT=(Bz-Az)*(Cx-Ax)-(Bx-Ax)*(Cz-Az);
double FS=(Bx-Ax)*(Cy-Ay)-(By-Ay)*(Cx-Ax);
double Dk=sqrt(FU*FU+FT*FT+FS*FS);
if(Dk <Gc){if(r.ov)return false;
continue;}
double nx=FU/Dk,ny=FT/Dk,nz=FS/Dk;
if(r.ov){if(r.onx*nx+r.ony*ny+r.onz*nz <At)return false;
double w=AJ?r.rpcW:Ej(FU,FT,FS);
double C0=AS(r.f,A5,
r.onx,r.ony,r.onz,nx,ny,nz);
Gq+=w*C0;
if(Lj){Iw+=r.nbpW*Ba(r.onx,r.ony,r.onz,nx,ny,nz);++Ho;}}
if(JE)for(int q=0;q<2;++q){if(!GG[q])continue;
double Mu=max(-1.0,min(1.0,Gr[q][0]*nx+Gr[q][1]*ny+Gr[q][2]*nz));
double d=1.0-Mu;
if(A5)d=AQ(A5,H6[q],d,
Gr[q][0],Gr[q][1],Gr[q][2],nx,ny,nz);
if(d<Ht[q])Ht[q]=d;}}
if(JE){double KN=0.0;
for(int q=0;q<2;++q){if(!GG[q])continue;
double AU=Ht[q];

KN+=JY[q]*AU;}
Gq+=AB*KN;}
KQ=Gq;
Kp=Ho> 0?Iw/Ho:1.0;
return true;}
double BB(int C7,int FL,double x,double y,double z){double c=C7==0?x:(C7==1?y:z);
return EO-FL*c;}
double Ev(int u,int v,int f0,int f1,
double px,double py,double pz){double A8=0.0;
for(int A6=0;A6 <2;++A6){int CP=A6==0?u:v;
for(int f:vf[CP]){if(!Ab[f]||f==f0||f==f1)continue;
int a=FA[f],b=FB[f],c=FC[f];
if(A6==1&&(a==u||b==u||c==u))continue;
auto gx=[&](int w){return(w==v||w==u)?px:X[w];};
auto gy=[&](int w){return(w==v||w==u)?py:Y[w];};
auto gz=[&](int w){return(w==v||w==u)?pz:Z[w];};
double w=(f <(int)Ae.size()&&Ae[f]> 0.0)?Ae[f]:0.0;
if(w <=0.0)continue;
double fc=0.0;
for(int C7=0;C7 <3;++C7)for(int FL=-1;FL <=1;FL+=2){double od=(BB(C7,FL,X[a],Y[a],Z[a])
+BB(C7,FL,X[b],Y[b],Z[b])
+BB(C7,FL,X[c],Y[c],Z[c]))/3.0;
double nd=(BB(C7,FL,gx(a),gy(a),gz(a))
+BB(C7,FL,gx(b),gy(b),gz(b))
+BB(C7,FL,gx(c),gy(c),gz(c)))/3.0;
double dd=nd-od;
fc+=dd*dd;}
A8+=w*fc/max(1e-12,CV);}}
return A8;}
vector<int> L8,LU,LT;
vector<double> JX,JW,JV,LS;
bool DJ;
void B9(int u,int v,int f0,int f1){L8.clear();LU.clear();LT.clear();
JX.clear();JW.clear();JV.clear();LS.clear();
DJ=false;
for(int A6=0;A6 <2;++A6){int CP=A6==0?u:v;
for(int f:vf[CP]){if(!Ab[f]||f==f0||f==f1)continue;
int a=FA[f],b=FB[f],c=FC[f];
if(A6==1&&(a==u||b==u||c==u))continue;
double onx,ony,onz;
if(!Af(X[a],Y[a],Z[a],X[b],Y[b],Z[b],X[c],Y[c],Z[c],onx,ony,onz))
continue;
int cu=(a==u)+(b==u)+(c==u),cv=(a==v)+(b==v)+(c==v);
if(cu&&cv)DJ=true;
L8.push_back(a);LU.push_back(b);LT.push_back(c);
JX.push_back(onx);JW.push_back(ony);JV.push_back(onz);
LS.push_back(AJ?((f <(int)Ae.size())?Ae[f]:0.0):0.0);}}
}
double BW(int u,int v,double cx,double cy,double cz){if(DJ)return 1e300;
double A8=0.0;
int n=(int)L8.size();
for(int i=0;i <n;++i){int a=L8[i],b=LU[i],c=LT[i];
double onx=JX[i],ony=JW[i],onz=JV[i];
double Ax=(a==u||a==v)?cx:X[a],Ay=(a==u||a==v)?cy:Y[a],Az=(a==u||a==v)?cz:Z[a];
double Bx=(b==u||b==v)?cx:X[b],By=(b==u||b==v)?cy:Y[b],Bz=(b==u||b==v)?cz:Z[b];
double Cx=(c==u||c==v)?cx:X[c],Cy=(c==u||c==v)?cy:Y[c],Cz=(c==u||c==v)?cz:Z[c];
double FU=(By-Ay)*(Cz-Az)-(Bz-Az)*(Cy-Ay);
double FT=(Bz-Az)*(Cx-Ax)-(Bx-Ax)*(Cz-Az);
double FS=(Bx-Ax)*(Cy-Ay)-(By-Ay)*(Cx-Ax);
double Dk=sqrt(FU*FU+FT*FT+FS*FS);
if(Dk <Gc)return 1e300;
double nx=FU/Dk,ny=FT/Dk,nz=FS/Dk;
if(onx*nx+ony*ny+onz*nz <At)return 1e300;
double w=AJ?LS[i]:Ej(FU,FT,FS);
double C0=1.0-(onx*FU/Dk+ony*FT/Dk+onz*FS/Dk);
A8+=w*C0;}
return A8;}
double DF(int u,int v,int f0,int f1,
double px,double py,double pz){double A8=0.0;
for(int A6=0;A6 <2;++A6){int CP=A6==0?u:v;
for(int f:vf[CP]){if(!Ab[f]||f==f0||f==f1)continue;
int a=FA[f],b=FB[f],c=FC[f];
if(A6==1&&(a==u||b==u||c==u))continue;
double onx,ony,onz;
if(!Af(X[a],Y[a],Z[a],X[b],Y[b],Z[b],X[c],Y[c],Z[c],onx,ony,onz))
continue;
auto gx=[&](int w){return(w==v||w==u)?px:X[w];};
auto gy=[&](int w){return(w==v||w==u)?py:Y[w];};
auto gz=[&](int w){return(w==v||w==u)?pz:Z[w];};
double nx,ny,nz,C0;
if(!Af(gx(a),gy(a),gz(a),gx(b),gy(b),gz(b),gx(c),gy(c),gz(c),nx,ny,nz))
C0=2.0;
else
C0=AS(f,BY,onx,ony,onz,nx,ny,nz);
double cov=(f <(int)Ae.size())?Ae[f]:0.0;
double w=pow((cov/D3)+1e-6,G0);
if(AR> 0.0&&f <(int)Av.size())w/=(1.0+AR*Av[f]);
A8+=w*C0;}}
if(Cg> 0.0)
A8+=Cg*AY(
u,v,f0,f1,px,py,pz,false,false,BY);
return A8;}
void Dc(){Av.assign(NF,0.0);
vector<array<double,3>> fn(NF);
for(int f=0;f <NF;++f){double nx,ny,nz;
if(Af(X[FA[f]],Y[FA[f]],Z[FA[f]],X[FB[f]],Y[FB[f]],Z[FB[f]],
X[FC[f]],Y[FC[f]],Z[FC[f]],nx,ny,nz))
fn[f]={nx,ny,nz};
else fn[f]={0.0,0.0,0.0};}
for(int f=0;f <NF;++f){int Hz[3]={FA[f],FB[f],FC[f]};
double Lx=0;int Eh=0;
for(int e=0;e <3;++e){int x=Hz[e],y=Hz[(e+1)%3];
int g=DG(f,x,y);
if(g>=0){Lx+=1.0-(fn[f][0]*fn[g][0]+fn[f][1]*fn[g][1]+fn[f][2]*fn[g][2]);
++Eh;}}
Av[f]=Eh?Lx/Eh:0.0;}}
void Dt(){CQ.assign(NF,{0.f,0.f,0.f});
vector<array<double,3>> fn(NF);
for(int f=0;f <NF;++f){double nx,ny,nz;
if(Af(X[FA[f]],Y[FA[f]],Z[FA[f]],X[FB[f]],Y[FB[f]],Z[FB[f]],
X[FC[f]],Y[FC[f]],Z[FC[f]],nx,ny,nz))fn[f]={nx,ny,nz};
else fn[f]={0.0,0.0,0.0};}
for(int f=0;f <NF;++f){int Hz[3]={FA[f],FB[f],FC[f]};
double ax=0,ay=0,az=0;
for(int e=0;e<3;++e){int x=Hz[e],y=Hz[(e+1)%3];
int g=DG(f,x,y);
if(g>=0){ax+=fn[g][0];ay+=fn[g][1];az+=fn[g][2];}}
double l=sqrt(ax*ax+ay*ay+az*az);
if(l>1e-30){CQ[f]={(float)(ax/l),(float)(ay/l),(float)(az/l)};}else CQ[f]={(float)fn[f][0],(float)fn[f][1],(float)fn[f][2]};}}
double DP(int u,int v,int f0,int f1,
double px,double py,double pz){double A8=0.0;
for(int A6=0;A6 <2;++A6){int CP=A6==0?u:v;
for(int f:vf[CP]){if(!Ab[f]||f==f0||f==f1)continue;
int a=FA[f],b=FB[f],c=FC[f];
if(A6==1&&(a==u||b==u||c==u))continue;
double onx,ony,onz;
if(!Af(X[a],Y[a],Z[a],X[b],Y[b],Z[b],X[c],Y[c],Z[c],onx,ony,onz))
continue;
auto gx=[&](int w){return(w==v||w==u)?px:X[w];};
auto gy=[&](int w){return(w==v||w==u)?py:Y[w];};
auto gz=[&](int w){return(w==v||w==u)?pz:Z[w];};
double nx,ny,nz,C0;
bool Mx=false;
if(!Af(gx(a),gy(a),gz(a),gx(b),gy(b),gz(b),gx(c),gy(c),gz(c),nx,ny,nz))
{C0=2.0;Mx=true;nx=onx;ny=ony;nz=onz;}else
C0=AS(f,BY,onx,ony,onz,nx,ny,nz);
if(Fb==1)C0=Mx?2.0:pow(2.0*max(0.0,C0),Gt);
double cov=(f <(int)Ae.size())?Ae[f]:0.0;
double w=pow((cov/D3)+1e-6,G0);
if(AR> 0.0&&f <(int)Av.size())w/=(1.0+AR*Av[f]);
double GX=1.0;
if(f <(int)CQ.size()){double mx=CQ[f][0],my=CQ[f][1],mz=CQ[f][2];
double Ll=Mx?2.0:(1.0-(nx*mx+ny*my+nz*mz));
double Lk=1.0-(onx*mx+ony*my+onz*mz);
double IE=Ll-Lk;
if(IE <0.0)IE=0.0;
GX=Ef+E6*IE;}
A8+=w*C0*GX;}}
if(Cg> 0.0)
A8+=Cg*AY(
u,v,f0,f1,px,py,pz,false,true,BY);
return A8;}
bool AD(int f,double&nx,double&ny,double&nz,
double&px,double&py,double&pz){if(f <0||f>=NF)return false;
int a=L3[f],b=L2[f],c=L1[f];
px=oX[a];py=oY[a];pz=oZ[a];
return Af(oX[a],oY[a],oZ[a],oX[b],oY[b],oZ[b],
oX[c],oY[c],oZ[c],nx,ny,nz);}
double DY(int v){double A8=0.0;
for(int f:vf[v]){if(!Ab[f])continue;
int a=FA[f],b=FB[f],c=FC[f];
if(a!=v&&b!=v&&c!=v)continue;
if(a==b||b==c||a==c)continue;
double nx,ny,nz,px,py,pz;
if(!AD(f,nx,ny,nz,px,py,pz))continue;
double w=(f <(int)Ae.size()&&Ae[f]> 0.0)?Ae[f]:1.0;
double da=(X[a]-px)*nx+(Y[a]-py)*ny+(Z[a]-pz)*nz;
double db=(X[b]-px)*nx+(Y[b]-py)*ny+(Z[b]-pz)*nz;
double dc=(X[c]-px)*nx+(Y[c]-py)*ny+(Z[c]-pz)*nz;
A8+=w*(da*da+db*db+dc*dc);}
return A8;}
bool FF(int v,double LG,double LF,double LE){for(int f:vf[v]){if(!Ab[f])continue;
int a=FA[f],b=FB[f],c=FC[f];
if(a!=v&&b!=v&&c!=v)continue;
if(a==b||b==c||a==c)return false;
auto gx=[&](int q){return q==v?LG:X[q];};
auto gy=[&](int q){return q==v?LF:Y[q];};
auto gz=[&](int q){return q==v?LE:Z[q];};
double Lq,Lp,Lo;
if(!Af(gx(a),gy(a),gz(a),gx(b),gy(b),gz(b),
gx(c),gy(c),gz(c),Lq,Lp,Lo))return false;
double onx,ony,onz,px,py,pz;
if(AD(f,onx,ony,onz,px,py,pz)){double It,Is,Ir;
if(Af(X[a],Y[a],Z[a],X[b],Y[b],Z[b],
X[c],Y[c],Z[c],It,Is,Ir)){double Hm=It*onx+Is*ony+Ir*onz;
double Hn=Lq*onx+Lp*ony+Lo*onz;
if(Hn+Fp <Hm)return false;}}
}
return true;}
void Fe(double Aq,vector<double>&Cf){if(!Cm)return;
vector<double> dx(NV),dy(NV),dz(NV),sw(NV);
for(int it=0;it <Bm;++it){memset(dx.data(),0,NV*sizeof(double));
memset(dy.data(),0,NV*sizeof(double));
memset(dz.data(),0,NV*sizeof(double));
memset(sw.data(),0,NV*sizeof(double));
for(int f=0;f <NF;++f)if(Ab[f]){int a=FA[f],b=FB[f],c=FC[f];
if(a==b||b==c||a==c)continue;
double nx,ny,nz,px,py,pz;
if(!AD(f,nx,ny,nz,px,py,pz))continue;
double w=(f <(int)Ae.size()&&Ae[f]> 0.0)?Ae[f]:1.0;
int vs[3]={a,b,c};
for(int k=0;k<3;++k){int v=vs[k];
double d=(X[v]-px)*nx+(Y[v]-py)*ny+(Z[v]-pz)*nz;
dx[v]+=-w*d*nx;dy[v]+=-w*d*ny;dz[v]+=-w*d*nz;sw[v]+=w;}}
for(int v=0;v <NV;++v)if(Ag[v]&&sw[v]> 0.0){double ox=X[v],oy=Y[v],oz=Z[v];
double bx=DY(v);
double sx=AN*dx[v]/sw[v];
double sy=AN*dy[v]/sw[v];
double sz=AN*dz[v]/sw[v];
double sl=sqrt(sx*sx+sy*sy+sz*sz);
double L9=max(0.0,0.985*Aq-Cf[v]);
if(sl <=1e-14||L9 <=1e-14)continue;
if(sl> L9){double t=L9/sl;sx*=t;sy*=t;sz*=t;sl=L9;}
double LG=ox+sx,LF=oy+sy,LE=oz+sz;
if(!FF(v,LG,LF,LE))continue;
X[v]=LG;Y[v]=LF;Z[v]=LE;
double ax=DY(v);
if(ax+Fp <bx)Cf[v]+=sl;
else{X[v]=ox;Y[v]=oy;Z[v]=oz;}}
}}
void Hh(double Aq,vector<double>&Cf){if(!MA)return;
const double GZ=0.985*Aq;
for(int it=0;it <G6;++it){int FV=0;
for(int v=0;v <NV;++v)if(Ag[v]){if(v>=(int)Cf.size())continue;
double Dx=GZ-Cf[v];
if(Dx <=1e-14)continue;
double Ea=0.0;int nf=0;
for(int f:vf[v]){if(!Ab[f])continue;
int a=FA[f],b=FB[f],c=FC[f];
if(a!=v&&b!=v&&c!=v)continue;
if(a==b||b==c||a==c)continue;
double onx,ony,onz,LZ,LX,LW;
if(!AD(f,onx,ony,onz,LZ,LX,LW))continue;
double nx,ny,nz;
if(!Af(X[a],Y[a],Z[a],X[b],Y[b],Z[b],X[c],Y[c],Z[c],nx,ny,nz))continue;
double w=(f <(int)Ae.size()&&Ae[f]> 0.0)?Ae[f]:1.0;
Ea+=w*Ba(onx,ony,onz,nx,ny,nz);
++nf;}
if(nf==0)continue;
Ea/=nf;
double cx=0,cy=0,cz=0;int nc=0;
vector<int> nb;As(v,nb);
for(int u:nb)if(Ag[u]){cx+=X[u];cy+=Y[u];cz+=Z[u];++nc;}
if(nc> 0){cx/=nc;cy/=nc;cz/=nc;}
double Kz[7][3]={
{CO,0,0},{-CO,0,0},{0,CO,0},{0,-CO,0},
{0,0,CO},{0,0,-CO},
{(cx-X[v])*0.5,(cy-Y[v])*0.5,(cz-Z[v])*0.5}};
double Ft=Ea;int JG=-1;
double IL=0.0,IK=0.0,IJ=0.0,Ha=0.0;
for(int di=0;di <7;++di){double LD=X[v],LC=Y[v],LB=Z[v];
double sx=Kz[di][0],sy=Kz[di][1],sz=Kz[di][2];
double DT=sqrt(sx*sx+sy*sy+sz*sz);
if(DT <=1e-14)continue;
if(DT> Dx){double t=Dx/DT;
sx*=t;sy*=t;sz*=t;DT=Dx;}
X[v]+=sx;Y[v]+=sy;Z[v]+=sz;
bool Ig=true;
for(int f:vf[v]){if(!Ab[f])continue;
int a=FA[f],b=FB[f],c=FC[f];
if(a!=v&&b!=v&&c!=v)continue;
if(a==b||b==c||a==c){Ig=false;break;}
auto gx=[&](int q){return q==v?X[v]:X[q];};
auto gy=[&](int q){return q==v?Y[v]:Y[q];};
auto gz=[&](int q){return q==v?Z[v]:Z[q];};
double Lq,Lp,Lo;
if(!Af(gx(a),gy(a),gz(a),gx(b),gy(b),gz(b),gx(c),gy(c),gz(c),Lq,Lp,Lo)){Ig=false;break;}
double Kj,Ki,Kh,N1,N0,Nz;
if(AD(f,Kj,Ki,Kh,N1,N0,Nz)){double Hn=Lq*Kj+Lp*Ki+Lo*Kh;
double NW=(b==v?LD:X[b]),NU=(b==v?LC:Y[b]),NT=(b==v?LB:Z[b]);
double NQ=(c==v?LD:X[c]),NP=(c==v?LC:Y[c]),NO=(c==v?LB:Z[c]);
double It,Is,Ir;
if(Af((a==v?LD:X[a]),(a==v?LC:Y[a]),(a==v?LB:Z[a]),NW,NU,NT,NQ,NP,NO,It,Is,Ir)){double Hm=It*Kj+Is*Ki+Ir*Kh;
if(Hn+1e-7 <Hm){Ig=false;break;}}
}}
double Eb=Ea;
if(Ig){Eb=0.0;int Md=0;
for(int f:vf[v]){if(!Ab[f])continue;
int a=FA[f],b=FB[f],c=FC[f];
if(a!=v&&b!=v&&c!=v)continue;
if(a==b||b==c||a==c)continue;
double onx,ony,onz,LZ,LX,LW;
if(!AD(f,onx,ony,onz,LZ,LX,LW))continue;
double nx,ny,nz;
if(!Af(X[a],Y[a],Z[a],X[b],Y[b],Z[b],X[c],Y[c],Z[c],nx,ny,nz))continue;
double w=(f <(int)Ae.size()&&Ae[f]> 0.0)?Ae[f]:1.0;
Eb+=w*Ba(onx,ony,onz,nx,ny,nz);
++Md;}
if(Md> 0)Eb/=Md;}
X[v]=LD;Y[v]=LC;Z[v]=LB;
if(Ig&&Eb <Ft-1e-12){Ft=Eb;JG=di;
IL=sx;IK=sy;IJ=sz;Ha=DT;}}
if(JG>=0&&Ft <Ea-1e-12){X[v]+=IL;Y[v]+=IK;Z[v]+=IJ;
Cf[v]+=Ha;
++FV;}}
if(FV==0)break;}}
double DA(int u,int v,int f0,int f1,double px,double py,double pz){double A8=0.0;int count=0;
for(int A6=0;A6 <2;++A6){int CP=A6==0?u:v;
for(int f:vf[CP]){if(!Ab[f]||f==f0||f==f1)continue;
int a=FA[f],b=FB[f],c=FC[f];
if(A6==1&&(a==u||b==u||c==u))continue;
double onx,ony,onz;
if(!Af(X[a],Y[a],Z[a],X[b],Y[b],Z[b],X[c],Y[c],Z[c],onx,ony,onz))continue;
auto gx=[&](int w){return(w==v||w==u)?px:X[w];};
auto gy=[&](int w){return(w==v||w==u)?py:Y[w];};
auto gz=[&](int w){return(w==v||w==u)?pz:Z[w];};
double nx,ny,nz;
if(!Af(gx(a),gy(a),gz(a),gx(b),gy(b),gz(b),gx(c),gy(c),gz(c),nx,ny,nz)){A8+=2.0;++count;continue;}
double w=(f <(int)Ae.size()&&Ae[f]> 0.0)?Ae[f]:1.0;
A8+=w*Ba(onx,ony,onz,nx,ny,nz);
++count;}}
return count> 0?A8/count:1.0;}
void Fo(){if(!Gy)return;
vector<double> onx(NV,0.0),ony(NV,0.0),onz(NV,0.0);
for(int f=0;f<NF;++f){int a=L3[f],b=L2[f],c=L1[f];
double nx=(oY[b]-oY[a])*(oZ[c]-oZ[a])-(oZ[b]-oZ[a])*(oY[c]-oY[a]);
double ny=(oZ[b]-oZ[a])*(oX[c]-oX[a])-(oX[b]-oX[a])*(oZ[c]-oZ[a]);
double nz=(oX[b]-oX[a])*(oY[c]-oY[a])-(oY[b]-oY[a])*(oX[c]-oX[a]);
onx[a]+=nx;ony[a]+=ny;onz[a]+=nz;onx[b]+=nx;ony[b]+=ny;onz[b]+=nz;onx[c]+=nx;ony[c]+=ny;onz[c]+=nz;}
auto Kx=[](int a,int b)->int64_t{return a>b?(((int64_t)b<<21)|(int64_t)a):(((int64_t)a<<21)|(int64_t)b);};
auto FX=[&](int a,int b,int c,double&nx,double&ny,double&nz)->double{
nx=(Y[b]-Y[a])*(Z[c]-Z[a])-(Z[b]-Z[a])*(Y[c]-Y[a]);
ny=(Z[b]-Z[a])*(X[c]-X[a])-(X[b]-X[a])*(Z[c]-Z[a]);
nz=(X[b]-X[a])*(Y[c]-Y[a])-(Y[b]-Y[a])*(X[c]-X[a]);
double l=sqrt(nx*nx+ny*ny+nz*nz);if(l>1e-30){nx/=l;ny/=l;nz/=l;}return l;};
for(int J3=0;J3<CW;++J3){unordered_map<int64_t,pair<int,int>> em;em.reserve((size_t)NF*3);
for(int f=0;f<NF;++f)if(Ab[f]){int v[3]={FA[f],FB[f],FC[f]};
if(v[0]==v[1]||v[1]==v[2]||v[0]==v[2])continue;
for(int e=0;e<3;++e){int64_t k=Kx(v[e],v[(e+1)%3]);
auto it=em.find(k);if(it==em.end())em[k]={f,-1};else if(it->second.second<0)it->second.second=f;}}
vector<char> ED(NF,0);
int JC=0;
for(auto&kv:em){int f0=kv.second.first,f1=kv.second.second;
if(f1<0||ED[f0]||ED[f1]||!Ab[f0]||!Ab[f1])continue;
int64_t k=kv.first;int u=(int)(k>>21),v=(int)(k&((1LL<<21)-1));
auto Gu=[&](int f)->int{int x=FA[f],y=FB[f],z=FC[f];if(x!=u&&x!=v)return x;if(y!=u&&y!=v)return y;return z;};
int a=Gu(f0),b=Gu(f1);
if(a==b||a==u||a==v||b==u||b==v)continue;
if(em.find(Kx(a,b))!=em.end())continue;
double tx=onx[u]+onx[v]+onx[a]+onx[b],ty=ony[u]+ony[v]+ony[a]+ony[b],tz=onz[u]+onz[v]+onz[a]+onz[b];
double tl=sqrt(tx*tx+ty*ty+tz*tz);if(tl<1e-30)continue;tx/=tl;ty/=tl;tz/=tl;
double Ne,Nd,Nc,Na,MZ,MY;
double l0=FX(FA[f0],FB[f0],FC[f0],Ne,Nd,Nc),l1=FX(FA[f1],FB[f1],FC[f1],Na,MZ,MY);
if(l0<1e-18||l1<1e-18)continue;
double LQ=(1.0-(Ne*tx+Nd*ty+Nc*tz))+(1.0-(Na*tx+MZ*ty+MY*tz));
double FU,FT,FS,LJ,LI,LH;
double la=FX(a,u,b,FU,FT,FS),lb=FX(a,b,v,LJ,LI,LH);
if(la<1e-18||lb<1e-18)continue;
int t0[3]={a,u,b},t1[3]={a,b,v};
if(FU*tx+FT*ty+FS*tz <0.0){swap(t0[1],t0[2]);FX(t0[0],t0[1],t0[2],FU,FT,FS);}
if(LJ*tx+LI*ty+LH*tz <0.0){swap(t1[1],t1[2]);FX(t1[0],t1[1],t1[2],LJ,LI,LH);}
double LP=(1.0-(FU*tx+FT*ty+FS*tz))+(1.0-(LJ*tx+LI*ty+LH*tz));
if(LP <LQ-EH){FA[f0]=t0[0];FB[f0]=t0[1];FC[f0]=t0[2];
FA[f1]=t1[0];FB[f1]=t1[1];FC[f1]=t1[2];
em[Kx(a,b)]={f0,f1};
ED[f0]=ED[f1]=1;++JC;}}
if(JC==0)break;}}
void Bi(){if((int)vf.size()!=NV)vf.resize(NV);
for(int i=0;i<NV;++i)vf[i].clear();
Q.assign(NV,array<double,10>{});
for(auto&q:Q)q.fill(0.0);
for(int f=0;f<NF;++f)if(Ab[f]){int a=FA[f],b=FB[f],c=FC[f];
if(a==b||b==c||a==c)continue;
vf[a].push_back(f);vf[b].push_back(f);vf[c].push_back(f);
auto q=B6(a,b,c);
double m=(f<(int)Bd.size()?Bd[f]:1.0);
if(m!=1.0)for(double&z:q)z*=m;
Ak(Q[a],q);Ak(Q[b],q);Ak(Q[c],q);}
A4.assign(NV,0);}
int Fs(int KG,double CZ,vector<double>&Cf){if(KG<=0)return 0;
Bi();
vector<vector<int>> cl(NV);
for(int i=0;i<NV;++i)cl[H2(i)].push_back(i);
vector<double> vc(NV,0.0);
auto E2=[&](){memset(vc.data(),0,NV*sizeof(double));
const int p=(int)Ae.size();
for(int f=0;f<NF;++f)if(Ab[f]){int a=FA[f],b=FB[f],c=FC[f];
double w=f<p?Ae[f]:0.0;
vc[a]+=w;vc[b]+=w;vc[c]+=w;}};
E2();
double av=0;int an=0;
auto Ds=[&](){av=0;an=0;for(int i=0;i<NV;++i)if(Ag[i]){av+=vc[i];++an;}
av=(an&&av>0)?av/an:1.0;};
Ds();
priority_queue<AH> pq;
const double HT=(A9*CZ)*(A9*CZ);
vector<int> nu,nv,nb,Bf,LR,E9;
vector<pair<int,int>> CD;
nu.reserve(32);nv.reserve(32);nb.reserve(32);Bf.reserve(64);LR.reserve(32);E9.reserve(32);CD.reserve(256);
auto Hx=[&](int a,int b,double x,double y,double z)->double{
double md=0.0;
for(int id:cl[a]){double dx=x-oX[id],dy=y-oY[id],dz=z-oZ[id];double d=dx*dx+dy*dy+dz*dz;if(d>md)md=d;if(md>HT)return md;}
for(int id:cl[b]){double dx=x-oX[id],dy=y-oY[id],dz=z-oZ[id];double d=dx*dx+dy*dy+dz*dz;if(d>md)md=d;if(md>HT)return md;}
return md;};
auto Ew=[&](int a,int b,double x,double y,double z)->double{
double md=0.0;
auto DV=[&](int Ii){for(int id:cl[Ii]){double dx=x-oX[id],dy=y-oY[id],dz=z-oZ[id];double AU=dx*dx+dy*dy+dz*dz;
for(int w:E9){dx=X[w]-oX[id];dy=Y[w]-oY[id];dz=Z[w]-oZ[id];double d=dx*dx+dy*dy+dz*dz;if(d<AU)AU=d;}
if(AU>md)md=AU;
if(md>HT)return false;}
return true;};
if(!DV(a)||!DV(b))return md;
double LV=1e300;
auto ER=[&](int Ii){for(int id:cl[Ii]){double dx=x-oX[id],dy=y-oY[id],dz=z-oZ[id];LV=min(LV,dx*dx+dy*dy+dz*dz);}};
ER(a);ER(b);for(int w:E9)ER(w);
return max(md,LV);};
auto Jm=[&](int a,int b,const int Gu[2])->bool{if(!Au)return AZ(a,b,Gu);
As(a,nu);As(b,nv);int Fr=0;
for(int x:nu)if(find(nv.begin(),nv.end(),x)!=nv.end()){++Fr;if(x!=Gu[0]&&x!=Gu[1])return false;}
return Fr==2;};
auto Kw=[&](int a,int b,double&bx,double&by,double&bz,double&bd,double&GJ)->bool{
int fs[2],op[2];if(Ar(a,b,fs,op)!=2||!Jm(a,b,op))return false;
if(Au){E9.clear();for(int w:nu)if(Ag[w]&&w!=a&&w!=b)E9.push_back(w);for(int w:nv)if(Ag[w]&&w!=a&&w!=b)E9.push_back(w);sort(E9.begin(),E9.end());E9.erase(unique(E9.begin(),E9.end()),E9.end());}
auto q=Q[a];Ak(q,Q[b]);double px,py,pz,qc;Am(a,b,q,px,py,pz,qc);
double mx=.5*(X[a]+X[b]),my=.5*(Y[a]+Y[b]),mz=.5*(Z[a]+Z[b]);
double c[16][3];int n=4;
c[0][0]=px;c[0][1]=py;c[0][2]=pz;c[1][0]=mx;c[1][1]=my;c[1][2]=mz;
c[2][0]=X[a];c[2][1]=Y[a];c[2][2]=Z[a];c[3][0]=X[b];c[3][1]=Y[b];c[3][2]=Z[b];
const double ts[3]={.25,.5,.75};
for(double t:ts){c[n][0]=px+t*(mx-px);c[n][1]=py+t*(my-py);c[n][2]=pz+t*(mz-pz);++n;}
for(double t:ts){c[n][0]=px+t*(X[a]-px);c[n][1]=py+t*(Y[a]-py);c[n][2]=pz+t*(Z[a]-pz);++n;}
for(double t:ts){c[n][0]=px+t*(X[b]-px);c[n][1]=py+t*(Y[b]-py);c[n][2]=pz+t*(Z[b]-pz);++n;}
double N2=At;At=AI;bool ok=false;double AU=1e300;
for(int k=0;k<n;++k){double x=c[k][0],y=c[k][1],z=c[k][2];double d2=Au?Ew(a,b,x,y,z):Hx(a,b,x,y,z);if(d2>HT)continue;
if(!BJ(a,b,fs[0],fs[1],x,y,z))continue;
double nc=Bh(a,b,fs[0],fs[1],x,y,z);if(nc>AO)continue;
double rp=BE(a,b,fs[0],fs[1],x,y,z);
double dp=Ev(a,b,fs[0],fs[1],x,y,z);
double qv=max(0.0,AG(q,x,y,z))/max(1e-18,CV);
double cr=.5*(vc[a]+vc[b])/max(1e-18,av);
double sc=rp+Bu*dp+.08*nc+.015*qv;
if(BS>0.0)sc+=BS*AY(a,b,fs[0],fs[1],x,y,z,true,false,A5);
if(cr>.35)sc*=1.0+.35*(cr-.35)*(cr-.35);
if(sc<AU){AU=sc;bx=x;by=y;bz=z;bd=sqrt(d2);GJ=sc;ok=true;}}
At=N2;return ok;};
auto FR=[&](int a,int b){if(a==b||!Ag[a]||!Ag[b])return;
if(a>b)swap(a,b);
int fs[2],op[2];if(Ar(a,b,fs,op)!=2)return;
if(Bl){double x,y,z,bd,r;if(Kw(a,b,x,y,z,bd,r)&&isfinite(r))pq.push({r,a,b,A4[a],A4[b]});return;}
auto q=Q[a];Ak(q,Q[b]);double x,y,z,qv;Am(a,b,q,x,y,z,qv);
double cr=.5*(vc[a]+vc[b])/max(1e-18,av);
double r=max(0.0,qv);
if(Br>0.0)r*=1.0+Br*DF(a,b,fs[0],fs[1],x,y,z);
r+=.01*CV*BE(a,b,fs[0],fs[1],x,y,z);
if(cr>.35)r*=1.0+.35*(cr-.35)*(cr-.35);
if(isfinite(r))pq.push({r,a,b,A4[a],A4[b]});};
auto D6=[&](){priority_queue<AH> e;pq.swap(e);
for(int f=0;f<NF;++f)if(Ab[f]){int v[3]={FA[f],FB[f],FC[f]};for(int k=0;k<3;++k)if(v[k]<v[(k+1)%3])FR(v[k],v[(k+1)%3]);}};
D6();
int Lm=0,GU=0,EF=0;
while(Lm<KG){if(pq.empty()){if(!Bk||GU)break;
int of=Gy,os=CW;double om=EH;
Gy=1;CW=1;EH=2e-5;Fo();
Gy=of;CW=os;EH=om;++GU;
Bi();E2();Ds();D6();continue;}
AH e;
double x=0,y=0,z=0,nd=0,r=1e300;
int u=-1,v=-1,fs[2],op[2];
if(Fw<=1){e=pq.top();pq.pop();u=e.u;v=e.v;
if(!Ag[u]||!Ag[v]||e.ver_u!=A4[u]||e.ver_v!=A4[v])continue;
if(!Kw(u,v,x,y,z,nd,r))continue;
if(Ar(u,v,fs,op)!=2)continue;
if(CK>=0&&Lm>=CK&&EF<Gm){++EF;continue;}}else{
struct HW{AH e;double x,y,z,nd,r;};
vector<HW> Ac;
vector<AH> Je;
Ac.reserve(Fw);
Je.reserve(Fw);
while(!pq.empty()&&(int)Ac.size()<Fw){AH ce=pq.top();pq.pop();
int cu=ce.u,cv=ce.v;
if(!Ag[cu]||!Ag[cv]||ce.ver_u!=A4[cu]||ce.ver_v!=A4[cv])continue;
double cx,cy,cz,KA,cr;
if(!Kw(cu,cv,cx,cy,cz,KA,cr))continue;
if(CK>=0&&Lm>=CK&&EF<Gm){++EF;continue;}
Ac.push_back({ce,cx,cy,cz,KA,cr});}
if(Ac.empty())continue;
int bi=0;
for(int i=1;i<(int)Ac.size();++i)if(Ac[i].r<Ac[bi].r)bi=i;
for(int i=0;i<(int)Ac.size();++i)if(i!=bi)pq.push(Ac[i].e);
e=Ac[bi].e;u=e.u;v=e.v;x=Ac[bi].x;y=Ac[bi].y;z=Ac[bi].z;nd=Ac[bi].nd;r=Ac[bi].r;
if(Ar(u,v,fs,op)!=2)continue;}
X[u]=x;Y[u]=y;Z[u]=z;Cf[u]=nd;
Ab[fs[0]]=Ab[fs[1]]=0;
for(int f:vf[v])if(Ab[f]){if(FA[f]==v)FA[f]=u;if(FB[f]==v)FB[f]=u;if(FC[f]==v)FC[f]=u;vf[u].push_back(f);}
Ag[v]=0;uf[v]=u;--A3;++Lm;
if(cl[u].size()<cl[v].size())cl[u].swap(cl[v]);
cl[u].insert(cl[u].end(),cl[v].begin(),cl[v].end());cl[v].clear();
if(vf[u].size()>64){vector<int> k;k.reserve(32);for(int f:vf[u])if(Ab[f])k.push_back(f);vf[u].swap(k);}
As(u,nb);Bf.clear();Bf.push_back(u);for(int w:nb)if(Ag[w])Bf.push_back(w);
for(int w:Bf)Ai(w);
++A4[v];for(int w:Bf)++A4[w];
for(int w:Bf){double cv=0;for(int f:vf[w])if(Ab[f]&&(FA[f]==w||FB[f]==w||FC[f]==w))cv+=f<(int)Ae.size()?Ae[f]:0.0;vc[w]=cv;}vc[v]=0;Ds();
CD.clear();for(int w:Bf){As(w,LR);for(int t:LR)if(Ag[t]&&w!=t)CD.emplace_back(min(w,t),max(w,t));}
sort(CD.begin(),CD.end());CD.erase(unique(CD.begin(),CD.end()),CD.end());for(auto p:CD)FR(p.first,p.second);}
return Lm;}
void Di(double CZ,vector<double>&Cf,long Hq){(void)Hq;if(!CX)return;
if(Co){double os=Bq;if(os<=0.0)Bq=1.0;BZ();Bq=os;}
if(Be>0)Fs(Be,CZ,Cf);}
struct C5{
double C[3],f[3],r[3],u[3];
double focal,cu,cv;};
struct Fz{
vector<float> z;
vector<int> fid;};
double Fd(double&x,double&y,double&z){double l=sqrt(x*x+y*y+z*z);
if(l> 1e-300){x/=l;y/=l;z/=l;}
return l;}
static C5 E3(int C7,int FL,int Al){C5 Bp{};
Bp.C[0]=Bp.C[1]=Bp.C[2]=0.0;
Bp.C[C7]=FL*F3;
Bp.f[0]=-Bp.C[0];Bp.f[1]=-Bp.C[1];Bp.f[2]=-Bp.C[2];
Fd(Bp.f[0],Bp.f[1],Bp.f[2]);
double up[3]={0,1,0};
if(fabs(Bp.f[1])> 0.9){up[0]=0;up[1]=0;up[2]=1;}
Bp.r[0]=Bp.f[1]*up[2]-Bp.f[2]*up[1];
Bp.r[1]=Bp.f[2]*up[0]-Bp.f[0]*up[2];
Bp.r[2]=Bp.f[0]*up[1]-Bp.f[1]*up[0];
Fd(Bp.r[0],Bp.r[1],Bp.r[2]);
Bp.u[0]=Bp.r[1]*Bp.f[2]-Bp.r[2]*Bp.f[1];
Bp.u[1]=Bp.r[2]*Bp.f[0]-Bp.r[0]*Bp.f[2];
Bp.u[2]=Bp.r[0]*Bp.f[1]-Bp.r[1]*Bp.f[0];
Bp.focal=HM*((double)Al/1024.0);
Bp.cu=Al/2.0;
Bp.cv=Al/2.0;
return Bp;}
bool D9(const C5&Bp,double x,double y,double z,
double&sx,double&sy,double&DO){double dx=x-Bp.C[0],dy=y-Bp.C[1],dz=z-Bp.C[2];
double xc=dx*Bp.r[0]+dy*Bp.r[1]+dz*Bp.r[2];
double yc=dx*Bp.u[0]+dy*Bp.u[1]+dz*Bp.u[2];
double zc=dx*Bp.f[0]+dy*Bp.f[1]+dz*Bp.f[2];
if(zc <1e-9)return false;
sx=Bp.focal*xc/zc+Bp.cu;
sy=Bp.focal*yc/zc+Bp.cv;
DO=zc;
return true;}
bool Cr(const C5&Bp,int v,double&sx,double&sy,double&DO){return D9(Bp,X[v],Y[v],Z[v],sx,sy,DO);}
void DW(vector<Fz>&Et,int Al){Et.assign(6,Fz{});
size_t N=(size_t)Al*Al;
for(int vi=0;vi<6;++vi){Et[vi].z.assign(N,3.4028235e38f);
Et[vi].fid.assign(N,-1);
C5 Bp=E3(Aw[vi][0],Aw[vi][1],Al);
for(int f=0;f<NF;++f)if(Ab[f]){int ia=FA[f],ib=FB[f],ic=FC[f];
if(ia==ib||ib==ic||ia==ic)continue;
double sx[3],sy[3],DO[3];
if(!Cr(Bp,ia,sx[0],sy[0],DO[0]))continue;
if(!Cr(Bp,ib,sx[1],sy[1],DO[1]))continue;
if(!Cr(Bp,ic,sx[2],sy[2],DO[2]))continue;
double De=(sx[1]-sx[0])*(sy[2]-sy[0])-(sx[2]-sx[0])*(sy[1]-sy[0]);
if(fabs(De)<1e-12)continue;
int x0=max(0,(int)floor(min({sx[0],sx[1],sx[2]})));
int x1=min(Al-1,(int)ceil(max({sx[0],sx[1],sx[2]})));
int y0=max(0,(int)floor(min({sy[0],sy[1],sy[2]})));
int y1=min(Al-1,(int)ceil(max({sy[0],sy[1],sy[2]})));
if(x0>x1||y0>y1)continue;
double F6=1.0/De;
double Jp=1.0/DO[0],Jo=1.0/DO[1],Jn=1.0/DO[2];
for(int y=y0;y<=y1;++y){double py=y+0.5;
for(int x=x0;x<=x1;++x){double px=x+0.5;
double w0=((sx[1]-px)*(sy[2]-py)-(sx[2]-px)*(sy[1]-py))*F6;
double w1=((sx[2]-px)*(sy[0]-py)-(sx[0]-px)*(sy[2]-py))*F6;
double w2=1.0-w0-w1;
if(w0<0||w1<0||w2<0)continue;
double zp=1.0/(w0*Jp+w1*Jo+w2*Jn);
size_t p=(size_t)y*Al+x;
if(zp <Et[vi].z[p]){Et[vi].z[p]=(float)zp;Et[vi].fid[p]=f;}}
}}
}}
int JM=1;
double F7=1e-3,Gd=0.02;
bool Hi(const C5&c,const vector<float>&zb,int Al,const double*A,const double*B,const double*D,double HF){double sx[3],sy[3],DO[3];const double*P[3]={A,B,D};
for(int k=0;k<3;++k)if(!D9(c,P[k][0],P[k][1],P[k][2],sx[k],sy[k],DO[k]))return false;
for(int k=0;k<3;++k){int x=(int)floor(sx[k]),y=(int)floor(sy[k]);
if(x<0||x>=Al||y<0||y>=Al)return false;
if(DO[k]<zb[(size_t)y*Al+x]+HF)return false;}
double De=(sx[1]-sx[0])*(sy[2]-sy[0])-(sx[2]-sx[0])*(sy[1]-sy[0]);
if(fabs(De)<1e-12)return true;
double F6=1.0/De;
int x0=max(0,(int)floor(min({sx[0],sx[1],sx[2]}))),x1=min(Al-1,(int)ceil(max({sx[0],sx[1],sx[2]})));
int y0=max(0,(int)floor(min({sy[0],sy[1],sy[2]}))),y1=min(Al-1,(int)ceil(max({sy[0],sy[1],sy[2]})));
double Jp=1.0/DO[0],Jo=1.0/DO[1],Jn=1.0/DO[2];
for(int y=y0;y<=y1;++y){double py=y+0.5;
for(int x=x0;x<=x1;++x){double px=x+0.5;
double w0=((sx[1]-px)*(sy[2]-py)-(sx[2]-px)*(sy[1]-py))*F6;
double w1=((sx[2]-px)*(sy[0]-py)-(sx[0]-px)*(sy[2]-py))*F6;
double w2=1.0-w0-w1;
if(w0<0||w1<0||w2<0)continue;
double zp=1.0/(w0*Jp+w1*Jo+w2*Jn);
if(zp<zb[(size_t)y*Al+x]+HF)return false;}}
return true;}
void G5(double CZ,vector<double>&Cf,double Aq){const int Al=1024;
double HF=F7*CZ,Jh=Gd*CZ;
vector<Fz> fr;
DW(fr,Al);
C5 Lr[6];
for(int vi=0;vi<6;++vi)Lr[vi]=E3(Aw[vi][0],Aw[vi][1],Al);
vector<vector<int>> Jb(NV);
for(int f=0;f<NF;++f)if(Ab[f]){int a=FA[f],b=FB[f],c=FC[f];
if(a==b||b==c||a==c)continue;
Jb[a].push_back(f);Jb[b].push_back(f);Jb[c].push_back(f);}
vector<int8_t> fh(NF,-1);
auto KJ=[&](const double*A,const double*B,const double*D)->bool{for(int vi=0;vi<6;++vi)if(!Hi(Lr[vi],fr[vi].z,Al,A,B,D,HF))return false;
return true;};
auto Ms=[&](int f)->bool{if(fh[f]>=0)return fh[f]==1;
double A[3]={X[FA[f]],Y[FA[f]],Z[FA[f]]},B[3]={X[FB[f]],Y[FB[f]],Z[FB[f]]},D[3]={X[FC[f]],Y[FC[f]],Z[FC[f]]};
bool h=KJ(A,B,D);
fh[f]=h?1:0;
return h;};
auto KW=[&](int v,vector<int>&nb){nb.clear();
for(int fi:Jb[v]){if(!Ab[fi])continue;
int iv[3]={FA[fi],FB[fi],FC[fi]};
for(int k=0;k<3;++k)if(iv[k]!=v)nb.push_back(iv[k]);}
sort(nb.begin(),nb.end());
nb.erase(unique(nb.begin(),nb.end()),nb.end());};
vector<int> nu,Iu,cm;
bool Kd=true;int Ih=0;
while(Kd&&Ih<50){Kd=false;++Ih;
for(int u=0;u<NV;++u){if(!Ag[u])continue;
bool MR=true;
for(int fi:Jb[u]){if(!Ab[fi])continue;if(!Ms(fi)){MR=false;break;}}
if(!MR)continue;
KW(u,nu);
for(int v:nu){if(!Ag[v])continue;
double dx=X[u]-X[v],dy=Y[u]-Y[v],dz=Z[u]-Z[v];
double d=sqrt(dx*dx+dy*dy+dz*dz);
if(d>Jh)continue;
if(Cf[u]+d>Aq)continue;
KW(v,Iu);
cm.clear();
set_intersection(nu.begin(),nu.end(),Iu.begin(),Iu.end(),back_inserter(cm));
int ef=0,op[2]={-1,-1};
for(int fi:Jb[u]){if(!Ab[fi])continue;
int iv[3]={FA[fi],FB[fi],FC[fi]};
bool hv=false;int o=-1;
for(int k=0;k<3;++k){if(iv[k]==v)hv=true;else if(iv[k]!=u)o=iv[k];}
if(hv){if(ef<2)op[ef]=o;++ef;}}
if(ef!=2||(int)cm.size()!=2||op[0]==op[1])continue;
bool ok=true;
for(int fi:Jb[u]){if(!Ab[fi])continue;
int iv[3]={FA[fi],FB[fi],FC[fi]};
bool hv=false;
for(int k=0;k<3;++k)if(iv[k]==v)hv=true;
if(hv)continue;
int w[3];
for(int k=0;k<3;++k)w[k]=(iv[k]==u)?v:iv[k];
double e1[3]={X[w[1]]-X[w[0]],Y[w[1]]-Y[w[0]],Z[w[1]]-Z[w[0]]};
double e2[3]={X[w[2]]-X[w[0]],Y[w[2]]-Y[w[0]],Z[w[2]]-Z[w[0]]};
double nx=e1[1]*e2[2]-e1[2]*e2[1],ny=e1[2]*e2[0]-e1[0]*e2[2],nz=e1[0]*e2[1]-e1[1]*e2[0];
if(nx*nx+ny*ny+nz*nz<1e-20){ok=false;break;}
double A[3]={X[w[0]],Y[w[0]],Z[w[0]]},B[3]={X[w[1]],Y[w[1]],Z[w[1]]},D[3]={X[w[2]],Y[w[2]],Z[w[2]]};
if(!KJ(A,B,D)){ok=false;break;}}
if(!ok)continue;
for(int fi:Jb[u]){if(!Ab[fi])continue;
int iv[3]={FA[fi],FB[fi],FC[fi]};
bool hv=false;
for(int k=0;k<3;++k)if(iv[k]==v)hv=true;
if(hv){Ab[fi]=0;continue;}
if(FA[fi]==u)FA[fi]=v;
if(FB[fi]==u)FB[fi]=v;
if(FC[fi]==u)FC[fi]=v;
fh[fi]=1;
Jb[v].push_back(fi);}
Ag[u]=0;
uf[u]=v;
--A3;
Cf[v]=max(Cf[v],Cf[u]+d);
Kd=true;
break;}}
}}
int HV;static double Bo=4.0,AX=0.0015,DQ=0.21;static int IG=1;static long G1=200000;
int BN;static double BO=0.90425,GR=1.04;
int HH;static double CN,C3;
static Dd Ad[6];static vector<float> Bs[6];
double B3[6];static long AT[6];
struct F1{double C[3],r[3],u[3],f[3],cu;};
static F1 F2[6];
const int Aa=5;int EB=32;
vector<vector<int>> CJ[6];
vector<int> Gl[6],Gj[6],Gk[6],Gi[6];
int BD;
vector<int> D1;static int Ep;
vector<double> J4,D7,Ka,CE,CF,Ja,JJ,JL,JK;
vector<float> F8,GB,GE,GC,HS;static vector<uint8_t> GD;
static vector<float> G9,HO,HR,HP;static vector<uint8_t> HQ;static int DM=0;
static vector<float> Jc[6],JR[6],JU[6],JS[6],KM[6];static vector<uint8_t> JT[6];
void Fj(int vi,int Al){F1&m=F2[vi];
int C7=Aw[vi][0],FL=Aw[vi][1];
m.C[0]=m.C[1]=m.C[2]=0;m.C[C7]=FL*F3;
double f[3]={-m.C[0],-m.C[1],-m.C[2]};
double fn=sqrt(f[0]*f[0]+f[1]*f[1]+f[2]*f[2]);f[0]/=fn;f[1]/=fn;f[2]/=fn;
double up[3]={0,1,0};if(fabs(f[1])>0.9){up[0]=0;up[1]=0;up[2]=1;}
double r[3]={f[1]*up[2]-f[2]*up[1],f[2]*up[0]-f[0]*up[2],f[0]*up[1]-f[1]*up[0]};
double rn=sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);r[0]/=rn;r[1]/=rn;r[2]/=rn;
double u[3]={r[1]*f[2]-r[2]*f[1],r[2]*f[0]-r[0]*f[2],r[0]*f[1]-r[1]*f[0]};
for(int k=0;k<3;k++){m.r[k]=r[k];m.u[k]=u[k];m.f[k]=f[k];}
m.cu=Al/2.0;}
bool CG(int vi,double x,double y,double z,double&sx,double&sy,double&zc){F1&m=F2[vi];
double dx=x-m.C[0],dy=y-m.C[1],dz=z-m.C[2];
double xc=dx*m.r[0]+dy*m.r[1]+dz*m.r[2],yc=dx*m.u[0]+dy*m.u[1]+dz*m.u[2];
zc=dx*m.f[0]+dy*m.f[1]+dz*m.f[2];
if(zc<1e-9)return false;
sx=HM*xc/zc+m.cu;sy=HM*yc/zc+m.cu;
return true;}
double Dh;
bool AV(int vi,const vector<int>&fs,int Al,int&x0,int&y0,int&x1,int&y1){double Fg=1e30,Ff=1e30,F5=-1e30,F4=-1e30;bool any=false;
Dh=1e30;
for(int fi:fs){if(!Ab[fi])continue;
int GS[3]={FA[fi],FB[fi],FC[fi]};
for(int k=0;k<3;k++){double sx,sy,zc;
if(!CG(vi,X[GS[k]],Y[GS[k]],Z[GS[k]],sx,sy,zc))continue;
if(zc<Dh)Dh=zc;
Fg=min(Fg,sx);F5=max(F5,sx);Ff=min(Ff,sy);F4=max(F4,sy);any=true;}}
if(!any)return false;
x0=max(0,(int)floor(Fg)-1);y0=max(0,(int)floor(Ff)-1);
x1=min(Al-1,(int)ceil(F5)+1);y1=min(Al-1,(int)ceil(F4)+1);
return x0<=x1&&y0<=y1;}
void E0(int vi,int Al,int fi,int&t0,int&t1,int&t2,int&t3){t0=1;t1=1;t2=0;t3=0;
if(!Ab[fi])return;
int ia=FA[fi],ib=FB[fi],ic=FC[fi];
if(ia==ib||ib==ic||ia==ic)return;
double Fg=1e30,Ff=1e30,F5=-1e30,F4=-1e30;int GS[3]={ia,ib,ic};
for(int k=0;k<3;k++){double sx,sy,zc;
if(!CG(vi,X[GS[k]],Y[GS[k]],Z[GS[k]],sx,sy,zc))return;
Fg=min(Fg,sx);F5=max(F5,sx);Ff=min(Ff,sy);F4=max(F4,sy);}
int x0=max(0,(int)floor(Fg)),y0=max(0,(int)floor(Ff)),x1=min(Al-1,(int)ceil(F5)),y1=min(Al-1,(int)ceil(F4));
if(x0>x1||y0>y1)return;
t0=x0/EB;t1=y0/EB;t2=x1/EB;t3=y1/EB;}
void BC(int Al){BD=(Al+EB-1)/EB;
if((int)D1.size()<NF)D1.assign(NF,0);
for(int vi=0;vi<6;vi++){CJ[vi].assign((size_t)BD*BD,{});
Gl[vi].assign(NF,1);Gj[vi].assign(NF,1);Gk[vi].assign(NF,0);Gi[vi].assign(NF,0);
for(int fi=0;fi<NF;fi++){if(!Ab[fi]&&!Dn(fi))continue;
int a,b,c,d;E0(vi,Al,fi,a,b,c,d);
Gl[vi][fi]=a;Gj[vi][fi]=b;Gk[vi][fi]=c;Gi[vi][fi]=d;
if(a>c)continue;
for(int ty=b;ty<=d;ty++)for(int tx=a;tx<=c;tx++)CJ[vi][(size_t)ty*BD+tx].push_back(fi);}}
}
void AM(int Al,const vector<int>&fs){for(int vi=0;vi<6;vi++)for(int fi:fs){int oa=Gl[vi][fi],ob=Gj[vi][fi],oc=Gk[vi][fi],od=Gi[vi][fi];
if(oa<=oc)for(int ty=ob;ty<=od;ty++)for(int tx=oa;tx<=oc;tx++){auto&cl=CJ[vi][(size_t)ty*BD+tx];
for(size_t k=0;k<cl.size();k++)if(cl[k]==fi){cl[k]=cl.back();cl.pop_back();break;}}
int a,b,c,d;E0(vi,Al,fi,a,b,c,d);
Gl[vi][fi]=a;Gj[vi][fi]=b;Gk[vi][fi]=c;Gi[vi][fi]=d;
if(a<=c)for(int ty=b;ty<=d;ty++)for(int tx=a;tx<=c;tx++)CJ[vi][(size_t)ty*BD+tx].push_back(fi);}}
void AF(int vi,int Al,int ES,int E1,int Hf,int Gp){Dd&B=Ad[vi];vector<float>&Zb=Bs[vi];
static vector<int> qf;qf.clear();++Ep;
for(int ty=E1/EB;ty<=Gp/EB;ty++)for(int tx=ES/EB;tx<=Hf/EB;tx++)
for(int fi:CJ[vi][(size_t)ty*BD+tx]){if(D1[fi]==Ep)continue;D1[fi]=Ep;qf.push_back(fi);}
sort(qf.begin(),qf.end());
const float M8=AC(127.5f);
for(int y=E1;y<=Gp;++y)for(int x=ES;x<=Hf;++x){size_t p=(size_t)y*Al+x;
B.nr[p]=M8;B.ng[p]=M8;B.nb[p]=M8;B.dp[p]=255.0f;B.cov[p]=0;Zb[p]=3.4028235e38f;}
for(int fi:qf){if(!Ab[fi]&&!Dn(fi))continue;
int ia=FA[fi],ib=FB[fi],ic=FC[fi];
if(ia==ib||ib==ic||ia==ic)continue;
double ax=X[ia],ay=Y[ia],az=Z[ia],bx=X[ib],by=Y[ib],bz=Z[ib],cx=X[ic],cy=Y[ic],cz=Z[ic];
double IC=bx-ax,IB=by-ay,IA=bz-az,I9=cx-ax,I8=cy-ay,I7=cz-az;
double Js=IB*I7-IA*I8,Jr=IA*I9-IC*I7,Jq=IC*I8-IB*I9;
double gl=sqrt(Js*Js+Jr*Jr+Jq*Jq);
if(gl<1e-30)continue;
Js/=gl;Jr/=gl;Jq/=gl;
float Rn=AC((float)((Js+1.0)*127.5)),Gn=AC((float)((Jr+1.0)*127.5)),Bn=AC((float)((Jq+1.0)*127.5));
double sx[3],sy[3],DO[3];bool ok=true;int GS[3]={ia,ib,ic};
for(int k=0;k<3;k++)if(!CG(vi,X[GS[k]],Y[GS[k]],Z[GS[k]],sx[k],sy[k],DO[k])){ok=false;break;}
if(!ok)continue;
double GN=min({sx[0],sx[1],sx[2]}),GP=max({sx[0],sx[1],sx[2]});
double GM=min({sy[0],sy[1],sy[2]}),GO=max({sy[0],sy[1],sy[2]});
int Mr=max(ES,(int)floor(GN)),Mq=min(Hf,(int)ceil(GP));
int MQ=max(E1,(int)floor(GM)),MP=min(Gp,(int)ceil(GO));
if(Mr>Mq||MQ>MP)continue;
double De=(sx[1]-sx[0])*(sy[2]-sy[0])-(sx[2]-sx[0])*(sy[1]-sy[0]);
if(fabs(De)<1e-12)continue;
double F6=1.0/De,Jp=1.0/DO[0],Jo=1.0/DO[1],Jn=1.0/DO[2];
for(int y=MQ;y<=MP;++y){double py=y+0.5;
double S2=sy[2]-py,S1=sy[1]-py,S0=sy[0]-py;size_t yr=(size_t)y*Al;
double Nm=(S1-S2)*F6,No=(sx[1]*S2-sx[2]*S1)*F6;
double Nl=(S2-S0)*F6,Nn=(sx[2]*S0-sx[0]*S2)*F6;
double NN=-(Nm+Nl),NS=1.0-No-Nn;
double M6=Mr+0.5,M7=Mq+0.5;bool Ly=false;
double NM[3]={Nm,Nl,NN},NR[3]={No,Nn,NS};
for(int _e=0;_e<3;_e++){double _m=NM[_e],_c=NR[_e];
if(_m>0.0){double _L=-_c/_m;if(_L>M6)M6=_L;}else if(_m<0.0){double _U=-_c/_m;if(_U<M7)M7=_U;}else if(_c<0.0){Ly=true;break;}}
int LM=Mr,LN=Mq;
if(Ly){LM=1;LN=0;}else{int _a=(int)floor(M6-0.5)-1,_b=(int)floor(M7-0.5)+1;if(_a>LM)LM=_a;if(_b<LN)LN=_b;}
for(int x=LM;x<=LN;++x){double px=x+0.5;
double w0=((sx[1]-px)*S2-(sx[2]-px)*S1)*F6;
double w1=((sx[2]-px)*S0-(sx[0]-px)*S2)*F6;
double w2=1.0-w0-w1;
if(w0<0||w1<0||w2<0)continue;
double zp=1.0/(w0*Jp+w1*Jo+w2*Jn);
size_t p=yr+x;
if(zp<Zb[p]){Zb[p]=(float)zp;B.dp[p]=AC((float)zp);B.nr[p]=Rn;B.ng[p]=Gn;B.nb[p]=Bn;B.cov[p]=1;}}}}
}
void Bw(int vi,int Al,int IZ,int IY,int JI,int JH,double&oS,long&oC,int Ca){oS=0;oC=0;
IZ=max(IZ,Aa);IY=max(IY,Aa);JI=min(JI,Al-1-Aa);JH=min(JH,Al-1-Aa);
if(IZ>JI||IY>JH)return;
int Kc=IZ-Aa,Kb=IY-Aa,Ma=JI+Aa,LY=JH+Aa;
int pw=Ma-Kc+1,ph=LY-Kb+1;size_t IW=(size_t)(pw+1)*(ph+1);
const double C1=6.5025,C2=58.5225,nn=121.0,L5=C1*nn*nn,L4=C2*nn*nn;
Dd&B=Ad[vi];const Dd&A=Bg[vi];
bool Kr=B0&&EX==vi&&EW==Kc&&EU==Kb&&EV==Ma&&ET==LY;
B0=0;
if(!Kr){EX=vi;EW=Kc;EU=Kb;EV=Ma;ET=LY;}
const vector<float>*ca[4]={&A.nr,&A.ng,&A.nb,&A.dp};
const vector<float>*cb[4]={&B.nr,&B.ng,&B.nb,&B.dp};
if(J4.size()<IW){J4.resize(IW);D7.resize(IW);Ka.resize(IW);CE.resize(IW);CF.resize(IW);}
if(!Kr)for(int ch=0;ch<4;ch++)if(Gs[ch].size()<IW){Gs[ch].resize(IW);GI[ch].resize(IW);}
int ww=JI-IZ+1,wh=JH-IY+1;size_t WG=(size_t)ww*wh;
if(Ja.size()<WG){Ja.resize(WG);JJ.resize(WG);JL.resize(WG);JK.resize(WG);}
vector<double>*Nt[4]={&Ja,&JJ,&JL,&JK};
size_t FM=(size_t)(pw+1);
for(int ch=0;ch<4;ch++){const vector<float>&Ax=*ca[ch];const vector<float>&Bx=*cb[ch];
vector<double>&PA=Gs[ch];vector<double>&HZ=GI[ch];
if(!Kr){
for(size_t i=0;i<FM;i++){PA[i]=0;D7[i]=0;HZ[i]=0;CE[i]=0;CF[i]=0;}
for(int ly=1;ly<=ph;ly++){size_t rp=(size_t)ly*FM,rm=(size_t)(ly-1)*FM;
PA[rp]=0;D7[rp]=0;HZ[rp]=0;CE[rp]=0;CF[rp]=0;
int gy=Kb+ly-1;size_t Mp=(size_t)gy*Al+Kc;
double KR=0,GT=0,In=0,Fk=0,GW=0;
for(int lx=1;lx<=pw;lx++){size_t gp=Mp+(lx-1);
double a=Ax[gp],b=Bx[gp];
KR+=a;PA[rp+lx]=PA[rm+lx]+KR;
GT+=b;D7[rp+lx]=D7[rm+lx]+GT;
In+=a*a;HZ[rp+lx]=HZ[rm+lx]+In;
Fk+=b*b;CE[rp+lx]=CE[rm+lx]+Fk;
GW+=a*b;CF[rp+lx]=CF[rm+lx]+GW;}}
}else{
for(size_t i=0;i<FM;i++){D7[i]=0;CE[i]=0;CF[i]=0;}
for(int ly=1;ly<=ph;ly++){size_t rp=(size_t)ly*FM,rm=(size_t)(ly-1)*FM;
D7[rp]=0;CE[rp]=0;CF[rp]=0;
int gy=Kb+ly-1;size_t Mp=(size_t)gy*Al+Kc;
double GT=0,Fk=0,GW=0;
for(int lx=1;lx<=pw;lx++){size_t gp=Mp+(lx-1);
double a=Ax[gp],b=Bx[gp];
GT+=b;D7[rp+lx]=D7[rm+lx]+GT;
Fk+=b*b;CE[rp+lx]=CE[rm+lx]+Fk;
GW+=a*b;CF[rp+lx]=CF[rm+lx]+GW;}}
}
auto II=[&](vector<double>&P,int r0,int c0,int r1,int c1)->double{return P[(size_t)(r1+1)*FM+(c1+1)]-P[(size_t)r0*FM+(c1+1)]-P[(size_t)(r1+1)*FM+c0]+P[(size_t)r0*FM+c0];};
for(int cy=IY;cy<=JH;cy+=Ca)for(int cx=IZ;cx<=JI;cx+=Ca){int r0=cy-Aa-Kb,c0=cx-Aa-Kc,r1=cy+Aa-Kb,c1=cx+Aa-Kc;
double sA=II(PA,r0,c0,r1,c1),sB=II(D7,r0,c0,r1,c1);
double Ny=II(HZ,r0,c0,r1,c1),Nw=II(CE,r0,c0,r1,c1);
double Nx=II(CF,r0,c0,r1,c1);
double MH=sA*sB,MG=sA*sA+sB*sB;
(*Nt[ch])[(size_t)(cy-IY)*ww+(cx-IZ)]=((2.0*MH+L5)*(2.0*(nn*Nx-MH)+L4))/((MG+L5)*(nn*(Ny+Nw)-MG+L4));}}
for(int cy=IY;cy<=JH;cy+=Ca)for(int cx=IZ;cx<=JI;cx+=Ca){size_t cp=(size_t)cy*Al+cx;
if(!(B.cov[cp]||A.cov[cp]))continue;
size_t wi=(size_t)(cy-IY)*ww+(cx-IZ);
oS+=(Ja[wi]+JJ[wi]+JL[wi])*(1.0/6.0)+0.5*JK[wi];++oC;}
if(Ca>1){double sc=(double)Ca*Ca;oS*=sc;oC=(long)llround((double)oC*sc);}}
void An(int Al){size_t N=(size_t)Al*Al;
for(int vi=0;vi<6;vi++){const Dd&R=Bg[vi];
for(size_t i=0;i<N;++i)Bt[i]=(R.cov[i]||Ad[vi].cov[i])?1:0;
double sr=Ao(R.nr,Ad[vi].nr,Bt,Al,Al,E8,FD,DD,DB,DC);
double sg=Ao(R.ng,Ad[vi].ng,Bt,Al,Al,E8,FD,DD,DB,DC);
double sb=Ao(R.nb,Ad[vi].nb,Bt,Al,Al,E8,FD,DD,DB,DC);
double sd=Ao(R.dp,Ad[vi].dp,Bt,Al,Al,E8,FD,DD,DB,DC);
long Eh=0;
for(int cy=Aa;cy<Al-Aa;++cy)for(int cx=Aa;cx<Al-Aa;++cx)if(Bt[(size_t)cy*Al+cx])++Eh;
AT[vi]=Eh;B3[vi]=Eh*(0.5*(sr+sg+sb)/3.0+0.5*sd);}}
double Ap(){double s=0;for(int vi=0;vi<6;vi++)s+=AT[vi]?B3[vi]/AT[vi]:1.0;
return s/6.0;}
clock_t HA=0;static double BX=1.2;
void Hy(double CZ,vector<double>&Cf,double Aq){if(!HV||A3<4)return;
if(C3>0.0&&(double)clock()>C3*CLOCKS_PER_SEC)return;
if(HH)DX=1;
Bi();
double D5=Bo*CLOCKS_PER_SEC;
int Al=1024;Dz=Al;
size_t N=(size_t)Al*Al;
{size_t IN=(size_t)(Al+1)*(Al+1);
if(E8.size()!=IN){E8.resize(IN);FD.resize(IN);DD.resize(IN);DB.resize(IN);DC.resize(IN);}
if(Bt.size()!=N)Bt.resize(N);}
if(Ij!=Al||GH!=DX)EA();
for(int vi=0;vi<6;vi++)Fj(vi,Al);
clock_t Ld=clock();
for(int vi=0;vi<6;vi++){AP(X,Y,Z,FA,FB,FC,Ab,NF,Aw[vi][0],Aw[vi][1],Al,Ad[vi].nr,Ad[vi].ng,Ad[vi].nb,Ad[vi].dp,Ad[vi].cov);
Bs[vi]=E5;}
An(Al);
double K4=(double)(clock()-Ld)/CLOCKS_PER_SEC;
BC(Al);
double Em=Ap();
vector<double> hX=X,hY=Y,hZ=Z,Ks=Cf;
vector<int> Mo=FA,Mn=FB,Mm=FC,Ml=uf,Lc=A4;
vector<uint8_t> Mg=Ab,Mf=Ag;
vector<vector<int>> Mk=vf;
int Hs=A3;
double Df=Em;
bool D4=BN&&Em>=BO;
clock_t t0=clock();
double DE=20.0*CLOCKS_PER_SEC;
if(BN){double Np=21.0;const char*al=getenv("ADAPT_LIMIT");
if(al){double v=atof(al);if(v>0.0)Np=v;}
double L0=0.965;const char*se=getenv("RSV_SAFE");if(se){double sv=atof(se);if(sv>0.0)L0=sv;}
double reserve=BX;
const char*rm=getenv("RSV_MODE");int GL=rm?atoi(rm):1;
if(GL==1)reserve=K4*1.4+0.3;
DE=(double)HA+Np*CLOCKS_PER_SEC*L0-reserve*CLOCKS_PER_SEC;
D5=DE-(double)t0;
}
double Ku=D4?BO:(Em+(CN>0.0?CN:1e-9));
int H3=0,FJ=0,IO=0,Id=0;
const int D0=getenv("FLIP_LOG")?1:0;
bool DR=false;
bool H1=false;
unordered_map<int64_t,pair<int,int>> K2;
auto FK=[&](){hX=X;hY=Y;hZ=Z;Mo=FA;Mn=FB;Mm=FC;Ml=uf;Lc=A4;
Mg=Ab;Mf=Ag;Mk=vf;Hs=A3;Ks=Cf;};
auto Eo=[&](){X=hX;Y=hY;Z=hZ;FA=Mo;FB=Mn;FC=Mm;uf=Ml;A4=Lc;
Ab=Mg;Ag=Mf;vf=Mk;A3=Hs;Cf=Ks;
for(int vi=0;vi<6;vi++){AP(X,Y,Z,FA,FB,FC,Ab,NF,Aw[vi][0],Aw[vi][1],Al,Ad[vi].nr,Ad[vi].ng,Ad[vi].nb,Ad[vi].dp,Ad[vi].cov);
Bs[vi]=E5;}
An(Al);BC(Al);Df=Ap();};
auto Dp=[&](int vi,int ES,int E1,int Hf,int Gp,int Ca,double&dS,long&dC){double oS,nS;long oC,nC;
if(DM){
int w=Hf-ES+1;
size_t sz=(size_t)w*(Gp-E1+1);
if(F8.size()<sz){F8.resize(sz);GB.resize(sz);GE.resize(sz);GC.resize(sz);HS.resize(sz);GD.resize(sz);}
if(G9.size()<sz){G9.resize(sz);HO.resize(sz);HR.resize(sz);HP.resize(sz);HQ.resize(sz);}
Dd&B=Ad[vi];vector<float>&Zb=Bs[vi];
for(int y=E1;y<=Gp;y++)for(int x=ES;x<=Hf;x++){size_t p=(size_t)y*Al+x,li=(size_t)(y-E1)*w+(x-ES);
F8[li]=B.nr[p];GB[li]=B.ng[p];GE[li]=B.nb[p];GC[li]=B.dp[p];HS[li]=Zb[p];GD[li]=B.cov[p];}
AF(vi,Al,ES,E1,Hf,Gp);
bool Hj=true;
for(int y=E1;y<=Gp&&Hj;y++){size_t p=(size_t)y*Al+ES,li=(size_t)(y-E1)*w;
if(memcmp(&F8[li],&B.nr[p],w*sizeof(float))||memcmp(&GB[li],&B.ng[p],w*sizeof(float))||memcmp(&GE[li],&B.nb[p],w*sizeof(float))||memcmp(&GC[li],&B.dp[p],w*sizeof(float))||memcmp(&GD[li],&B.cov[p],w*sizeof(uint8_t)))Hj=false;}
if(Hj){dS=0.0;dC=0;}
else{
for(int y=E1;y<=Gp;y++){size_t p=(size_t)y*Al+ES,li=(size_t)(y-E1)*w;
memcpy(&G9[li],&B.nr[p],w*sizeof(float));
memcpy(&HO[li],&B.ng[p],w*sizeof(float));
memcpy(&HR[li],&B.nb[p],w*sizeof(float));
memcpy(&HP[li],&B.dp[p],w*sizeof(float));
memcpy(&HQ[li],&B.cov[p],w*sizeof(uint8_t));
memcpy(&B.nr[p],&F8[li],w*sizeof(float));
memcpy(&B.ng[p],&GB[li],w*sizeof(float));
memcpy(&B.nb[p],&GE[li],w*sizeof(float));
memcpy(&B.dp[p],&GC[li],w*sizeof(float));
memcpy(&B.cov[p],&GD[li],w*sizeof(uint8_t));}
Bw(vi,Al,ES-Aa,E1-Aa,Hf+Aa,Gp+Aa,oS,oC,Ca);
for(int y=E1;y<=Gp;y++){size_t p=(size_t)y*Al+ES,li=(size_t)(y-E1)*w;
memcpy(&B.nr[p],&G9[li],w*sizeof(float));
memcpy(&B.ng[p],&HO[li],w*sizeof(float));
memcpy(&B.nb[p],&HR[li],w*sizeof(float));
memcpy(&B.dp[p],&HP[li],w*sizeof(float));
memcpy(&B.cov[p],&HQ[li],w*sizeof(uint8_t));}
B0=1;
Bw(vi,Al,ES-Aa,E1-Aa,Hf+Aa,Gp+Aa,nS,nC,Ca);
dS=nS-oS;dC=nC-oC;}
for(int y=E1;y<=Gp;y++)for(int x=ES;x<=Hf;x++){size_t p=(size_t)y*Al+x,li=(size_t)(y-E1)*w+(x-ES);
B.nr[p]=F8[li];B.ng[p]=GB[li];B.nb[p]=GE[li];B.dp[p]=GC[li];Zb[p]=HS[li];B.cov[p]=GD[li];}
return;}
Bw(vi,Al,ES-Aa,E1-Aa,Hf+Aa,Gp+Aa,oS,oC,Ca);
int w=Hf-ES+1;
size_t sz=(size_t)w*(Gp-E1+1);
if(F8.size()<sz){F8.resize(sz);GB.resize(sz);GE.resize(sz);GC.resize(sz);HS.resize(sz);GD.resize(sz);}
Dd&B=Ad[vi];vector<float>&Zb=Bs[vi];
for(int y=E1;y<=Gp;y++)for(int x=ES;x<=Hf;x++){size_t p=(size_t)y*Al+x,li=(size_t)(y-E1)*w+(x-ES);
F8[li]=B.nr[p];GB[li]=B.ng[p];GE[li]=B.nb[p];GC[li]=B.dp[p];HS[li]=Zb[p];GD[li]=B.cov[p];}
AF(vi,Al,ES,E1,Hf,Gp);
B0=1;
Bw(vi,Al,ES-Aa,E1-Aa,Hf+Aa,Gp+Aa,nS,nC,Ca);
for(int y=E1;y<=Gp;y++)for(int x=ES;x<=Hf;x++){size_t p=(size_t)y*Al+x,li=(size_t)(y-E1)*w+(x-ES);
B.nr[p]=F8[li];B.ng[p]=GB[li];B.nb[p]=GE[li];B.dp[p]=GC[li];Zb[p]=HS[li];B.cov[p]=GD[li];}
dS=nS-oS;dC=nC-oC;};
auto GF=[&](int vi,int x0,int y0,int x1,int y1,double Me)->bool{
vector<float>&Zb=Bs[vi];
double Nv=Me-1e-5;
for(int y=y0;y<=y1;y++){size_t rw=(size_t)y*Al;
for(int x=x0;x<=x1;x++)if((double)Zb[rw+x]>=Nv)return false;}
return true;};
auto JF=[&](double GY)->int{
int FV=0;
double DU=AX*CZ;
vector<int> G8;G8.reserve(A3);
{vector<double> vw(NV,0.0);
for(int f=0;f<NF;f++)if(Ab[f]){int a=FA[f],b=FB[f],c=FC[f];
if(a==b||b==c||a==c)continue;
double ux=X[b]-X[a],uy=Y[b]-Y[a],uz=Z[b]-Z[a],vx=X[c]-X[a],vy=Y[c]-Y[a],vz=Z[c]-Z[a];
double MW=uy*vz-uz*vy,MV=uz*vx-ux*vz,MU=ux*vy-uy*vx;
double ar=0.5*sqrt(MW*MW+MV*MV+MU*MU);
vw[a]+=ar;vw[b]+=ar;vw[c]+=ar;}
for(int v=0;v<NV;v++)if(Ag[v])G8.push_back(v);
sort(G8.begin(),G8.end(),[&](int a,int b){return vw[a]>vw[b];});}
double EK[6][3];
vector<int> iv;iv.reserve(16);
bool KL=false;
for(int ps=0;ps<3&&!KL;ps++){int mv=0;
for(int v:G8){if((double)(clock()-t0)>GY){KL=true;break;}
if(!Ag[v])continue;
if(Cf[v]+DU>0.985*Aq)continue;
iv.clear();
double nx=0,ny=0,nz=0;
for(int f:vf[v]){if(!Ab[f])continue;
int a=FA[f],b=FB[f],c=FC[f];
if(a!=v&&b!=v&&c!=v)continue;
if(a==b||b==c||a==c)continue;
iv.push_back(f);
double fx,fy,fz;
if(Af(X[a],Y[a],Z[a],X[b],Y[b],Z[b],X[c],Y[c],Z[c],fx,fy,fz)){nx+=fx;ny+=fy;nz+=fz;}}
if(iv.empty())continue;
double nl=sqrt(nx*nx+ny*ny+nz*nz);
if(nl<1e-12)continue;
nx/=nl;ny/=nl;nz/=nl;
double ax=fabs(nx)<0.9?1.0:0.0,ay=1.0-ax;
double JQ=ny*0.0-nz*ay,JP=nz*ax-nx*0.0,JO=nx*ay-ny*ax;
double tl=sqrt(JQ*JQ+JP*JP+JO*JO);
if(tl<1e-12)continue;
JQ/=tl;JP/=tl;JO/=tl;
double MD=ny*JO-nz*JP,MC=nz*JQ-nx*JO,MB=nx*JP-ny*JQ;
EK[0][0]=nx;EK[0][1]=ny;EK[0][2]=nz;
EK[1][0]=-nx;EK[1][1]=-ny;EK[1][2]=-nz;
EK[2][0]=JQ;EK[2][1]=JP;EK[2][2]=JO;
EK[3][0]=-JQ;EK[3][1]=-JP;EK[3][2]=-JO;
EK[4][0]=MD;EK[4][1]=MC;EK[4][2]=MB;
EK[5][0]=-MD;EK[5][1]=-MC;EK[5][2]=-MB;
double ox=X[v],oy=Y[v],oz=Z[v];
double Cd=Df;int bd=-1;
double HX[6];long HY[6];int bB[6][4];bool bU[6];
int ob[6][4];bool C6[6];
for(int vi=0;vi<6;vi++)C6[vi]=AV(vi,iv,Al,ob[vi][0],ob[vi][1],ob[vi][2],ob[vi][3]);
for(int d=0;d<6;d++){double px=ox+EK[d][0]*DU,py=oy+EK[d][1]*DU,pz=oz+EK[d][2]*DU;
bool MM=true;
for(int f:iv){int a=FA[f],b=FB[f],c=FC[f];
auto gx=[&](int q){return q==v?px:X[q];};
auto gy=[&](int q){return q==v?py:Y[q];};
auto gz=[&](int q){return q==v?pz:Z[q];};
double qx,qy,qz;
if(!Af(gx(a),gy(a),gz(a),gx(b),gy(b),gz(b),gx(c),gy(c),gz(c),qx,qy,qz)){MM=false;break;}}
if(!MM)continue;
X[v]=px;Y[v]=py;Z[v]=pz;
int bb[6][4];bool KI[6]={false,false,false,false,false,false};
for(int vi=0;vi<6;vi++){int c0,d0,c1,d1;
bool hb=AV(vi,iv,Al,c0,d0,c1,d1);
if(!C6[vi]&&!hb)continue;
if(!hb){bb[vi][0]=ob[vi][0];bb[vi][1]=ob[vi][1];bb[vi][2]=ob[vi][2];bb[vi][3]=ob[vi][3];KI[vi]=true;}
else if(!C6[vi]){bb[vi][0]=c0;bb[vi][1]=d0;bb[vi][2]=c1;bb[vi][3]=d1;KI[vi]=true;}
else{int ND=min(ob[vi][0],c0),NB=min(ob[vi][1],d0),NC=max(ob[vi][2],c1),NA=max(ob[vi][3],d1);
bb[vi][0]=ND;bb[vi][1]=NB;bb[vi][2]=NC;bb[vi][3]=NA;KI[vi]=true;}}
double ns=0;
double ds[6];long dc[6];
for(int vi=0;vi<6;vi++){double vs=B3[vi],vc=(double)AT[vi];
if(KI[vi]){long De=(long)(bb[vi][2]-bb[vi][0]+1+2*Aa)*(bb[vi][3]-bb[vi][1]+1+2*Aa);
int st=(De>G1)?IG:1;
Dp(vi,bb[vi][0],bb[vi][1],bb[vi][2],bb[vi][3],st,ds[vi],dc[vi]);
vs+=ds[vi];vc+=(double)dc[vi];}else{ds[vi]=0;dc[vi]=0;}
ns+=vc>0?vs/vc:1.0;}
ns/=6.0;
X[v]=ox;Y[v]=oy;Z[v]=oz;
if(ns>Cd){Cd=ns;bd=d;
for(int vi=0;vi<6;vi++){bU[vi]=KI[vi];HX[vi]=ds[vi];HY[vi]=dc[vi];for(int q2=0;q2<4;q2++)bB[vi][q2]=bb[vi][q2];}}
}
if(bd>=0&&Cd>Df+1e-9){X[v]=ox+EK[bd][0]*DU;Y[v]=oy+EK[bd][1]*DU;Z[v]=oz+EK[bd][2]*DU;
for(int vi=0;vi<6;vi++)if(bU[vi]){AF(vi,Al,bB[vi][0],bB[vi][1],bB[vi][2],bB[vi][3]);B3[vi]+=HX[vi];AT[vi]+=HY[vi];}
AM(Al,iv);
Cf[v]+=DU;
Df=Cd;++mv;++FV;
if(FV%128==0){An(Al);Df=Ap();}}
}
if(mv==0)break;}
return FV;};
vector<vector<int>> FE;
J0 Ky;
double JD=0.070*CZ;
double Hv=JD*JD;
double GV=0.0;
long Lh=0,I6=0,HU=0,Jt=0;
double HE=0.0,HD=0.0;
auto Ju=[&](int u,int v,const int G4[2],double px,double py,double pz,double&KT,double&KS)->bool{
(void)G4;
double Gw=0.0;
auto GA=[&](int Ii)->bool{for(int Kl:FE[Ii]){double dx=oX[Kl]-px,dy=oY[Kl]-py,dz=oZ[Kl]-pz;
double d2=dx*dx+dy*dy+dz*dz;if(d2>Gw)Gw=d2;if(Gw>=Hv)return false;}
return true;};
if(!GA(u))return false;
if(!GA(v))return false;
double ME=Ky.nearest2(px,py,pz,Hv);
if(ME>=Hv)return false;
KT=sqrt(Gw);KS=sqrt(ME);return true;};
auto EJ=[&](int KX,bool BL)->int{
int AK=0;
for(int v2=0;v2<NV;v2++)if(Ag[v2])Ai(v2);
priority_queue<AH> cq;
auto IF=[&](int a,int b){if(a==b)return;
if(a>b)swap(a,b);
if(!Ag[a]||!Ag[b])return;
if(!A2.empty()&&(A2[a]||A2[b]))return;
int G4[2],K5[2];
if(Ar(a,b,G4,K5)!=2)return;
array<double,10> qq=Q[a];Ak(qq,Q[b]);
double px,py,pz,c;
Am(a,b,qq,px,py,pz,c);
if(!isfinite(c))return;
cq.push({c,a,b,A4[a],A4[b]});};
for(int f=0;f<NF;f++)if(Ab[f]){int Hz[3]={FA[f],FB[f],FC[f]};
for(int e2=0;e2<3;e2++){int a=Hz[e2],b=Hz[(e2+1)%3];if(a<b)IF(a,b);}}
vector<int> K6,C9;
vector<pair<int,int>> FQ;
while(!cq.empty()&&A3>4&&AK<KX){if((double)(clock()-t0)>D5)break;
AH ed=cq.top();cq.pop();
int u=ed.u,v=ed.v;
if(!Ag[u]||!Ag[v]||ed.ver_u!=A4[u]||ed.ver_v!=A4[v])continue;
if(!A2.empty()&&(A2[u]||A2[v]))continue;
int64_t bk=(int64_t)u*NV+v;
{auto it=K2.find(bk);
if(it!=K2.end()&&it->second.first==A4[u]&&it->second.second==A4[v])continue;}
int G4[2],K5[2];
if(Ar(u,v,G4,K5)!=2)continue;
if(!AZ(u,v,K5))continue;
array<double,10> qq=Q[u];Ak(qq,Q[v]);
double qx,qy,qz,qc;
Am(u,v,qq,qx,qy,qz,qc);
double MK=X[u],MJ=Y[u],MI=Z[u];
double KA[4][3]={{qx,qy,qz},{0.5*(X[u]+X[v]),0.5*(Y[u]+Y[v]),0.5*(Z[u]+Z[v])},{X[u],Y[u],Z[u]},{X[v],Y[v],Z[v]}};
double vp[4][3],L6[4];int Iu=0;
double KU=At;At=DQ;
if(BL)++Jt;
double IH=0.0;
for(int ci=0;ci<4;ci++){double px=KA[ci][0],py=KA[ci][1],pz=KA[ci][2];
double du=sqrt((px-X[u])*(px-X[u])+(py-Y[u])*(py-Y[u])+(pz-Z[u])*(pz-Z[u]));
double NH=sqrt((px-X[v])*(px-X[v])+(py-Y[v])*(py-Y[v])+(pz-Z[v])*(pz-Z[v]));
double dn=max(Cf[u]+du,Cf[v]+NH)*A1;
double Ic=dn;
if(!BL){if(dn>Aq*0.98)continue;}
else{double Lg,Lf;if(!Ju(u,v,G4,px,py,pz,Lg,Lf))continue;Ic=Lg;IH=Lf;}
if(!BJ(u,v,G4[0],G4[1],px,py,pz))continue;
vp[Iu][0]=px;vp[Iu][1]=py;vp[Iu][2]=pz;L6[Iu]=Ic;++Iu;
if(D4)break;}
At=KU;
if(Iu==0){if(BL)++I6;continue;}
if(BL)GV=IH;
C9.clear();
for(int f:vf[u])if(Ab[f]&&(FA[f]==u||FB[f]==u||FC[f]==u))C9.push_back(f);
for(int f:vf[v])if(Ab[f]&&(FA[f]==v||FB[f]==v||FC[f]==v))C9.push_back(f);
sort(C9.begin(),C9.end());
C9.erase(unique(C9.begin(),C9.end()),C9.end());
int ob[6][4];bool C6[6];double ML[6];
for(int vi=0;vi<6;vi++){C6[vi]=AV(vi,C9,Al,ob[vi][0],ob[vi][1],ob[vi][2],ob[vi][3]);ML[vi]=Dh;}
Ab[G4[0]]=0;Ab[G4[1]]=0;
FQ.clear();
for(int f:vf[v]){if(!Ab[f])continue;
if(FA[f]==v){FA[f]=u;FQ.push_back({f,0});}
if(FB[f]==v){FB[f]=u;FQ.push_back({f,1});}
if(FC[f]==v){FC[f]=u;FQ.push_back({f,2});}}
double HX[6];long HY[6];int bB[6][4];bool bU[6];
if(D4){X[u]=vp[0][0];Y[u]=vp[0][1];Z[u]=vp[0][2];
double ns=0;
for(int vi=0;vi<6;vi++){int c0,d0,c1,d1;
bool hb=AV(vi,C9,Al,c0,d0,c1,d1);
double N9=hb?Dh:1e30;
bU[vi]=C6[vi]||hb;
if(!bU[vi]){HX[vi]=0;HY[vi]=0;ns+=AT[vi]?(double)B3[vi]/AT[vi]:1.0;continue;}
int X0,Y0,X1,Y1;
if(C6[vi]&&hb){X0=min(ob[vi][0],c0);Y0=min(ob[vi][1],d0);X1=max(ob[vi][2],c1);Y1=max(ob[vi][3],d1);}else if(C6[vi]){X0=ob[vi][0];Y0=ob[vi][1];X1=ob[vi][2];Y1=ob[vi][3];}else{X0=c0;Y0=d0;X1=c1;Y1=d1;}
if(GF(vi,X0,Y0,X1,Y1,min(C6[vi]?ML[vi]:1e30,N9))){bU[vi]=false;HX[vi]=0;HY[vi]=0;
ns+=AT[vi]?(double)B3[vi]/AT[vi]:1.0;continue;}
bB[vi][0]=X0;bB[vi][1]=Y0;bB[vi][2]=X1;bB[vi][3]=Y1;
double oS,nS;long oC,nC;
if(DM){
Dd&Bv=Ad[vi];vector<float>&Lz=Bs[vi];
int w=X1-X0+1,h=Y1-Y0+1;size_t sz=(size_t)w*h;
if(Jc[vi].size()<sz){Jc[vi].resize(sz);JR[vi].resize(sz);JU[vi].resize(sz);JS[vi].resize(sz);KM[vi].resize(sz);JT[vi].resize(sz);}
if(G9.size()<sz){G9.resize(sz);HO.resize(sz);HR.resize(sz);HP.resize(sz);HQ.resize(sz);}
for(int yy=Y0;yy<=Y1;yy++){size_t sp=(size_t)yy*Al+X0,dp=(size_t)(yy-Y0)*w;
memcpy(&Jc[vi][dp],&Bv.nr[sp],w*sizeof(float));
memcpy(&JR[vi][dp],&Bv.ng[sp],w*sizeof(float));
memcpy(&JU[vi][dp],&Bv.nb[sp],w*sizeof(float));
memcpy(&JS[vi][dp],&Bv.dp[sp],w*sizeof(float));
memcpy(&KM[vi][dp],&Lz[sp],w*sizeof(float));
memcpy(&JT[vi][dp],&Bv.cov[sp],w*sizeof(uint8_t));}
AF(vi,Al,X0,Y0,X1,Y1);
bool Hj=true;
for(int yy=Y0;yy<=Y1&&Hj;yy++){size_t sp=(size_t)yy*Al+X0,dp=(size_t)(yy-Y0)*w;
if(memcmp(&Jc[vi][dp],&Bv.nr[sp],w*sizeof(float))||memcmp(&JR[vi][dp],&Bv.ng[sp],w*sizeof(float))||memcmp(&JU[vi][dp],&Bv.nb[sp],w*sizeof(float))||memcmp(&JS[vi][dp],&Bv.dp[sp],w*sizeof(float))||memcmp(&JT[vi][dp],&Bv.cov[sp],w*sizeof(uint8_t)))Hj=false;}
if(Hj){HX[vi]=0.0;HY[vi]=0;}
else{
for(int yy=Y0;yy<=Y1;yy++){size_t sp=(size_t)yy*Al+X0,dp=(size_t)(yy-Y0)*w;
memcpy(&G9[dp],&Bv.nr[sp],w*sizeof(float));
memcpy(&HO[dp],&Bv.ng[sp],w*sizeof(float));
memcpy(&HR[dp],&Bv.nb[sp],w*sizeof(float));
memcpy(&HP[dp],&Bv.dp[sp],w*sizeof(float));
memcpy(&HQ[dp],&Bv.cov[sp],w*sizeof(uint8_t));
memcpy(&Bv.nr[sp],&Jc[vi][dp],w*sizeof(float));
memcpy(&Bv.ng[sp],&JR[vi][dp],w*sizeof(float));
memcpy(&Bv.nb[sp],&JU[vi][dp],w*sizeof(float));
memcpy(&Bv.dp[sp],&JS[vi][dp],w*sizeof(float));
memcpy(&Bv.cov[sp],&JT[vi][dp],w*sizeof(uint8_t));}
Bw(vi,Al,X0-Aa,Y0-Aa,X1+Aa,Y1+Aa,oS,oC,1);
for(int yy=Y0;yy<=Y1;yy++){size_t sp=(size_t)yy*Al+X0,dp=(size_t)(yy-Y0)*w;
memcpy(&Bv.nr[sp],&G9[dp],w*sizeof(float));
memcpy(&Bv.ng[sp],&HO[dp],w*sizeof(float));
memcpy(&Bv.nb[sp],&HR[dp],w*sizeof(float));
memcpy(&Bv.dp[sp],&HP[dp],w*sizeof(float));
memcpy(&Bv.cov[sp],&HQ[dp],w*sizeof(uint8_t));}
B0=1;
Bw(vi,Al,X0-Aa,Y0-Aa,X1+Aa,Y1+Aa,nS,nC,1);
HX[vi]=nS-oS;HY[vi]=nC-oC;}}
else{
Bw(vi,Al,X0-Aa,Y0-Aa,X1+Aa,Y1+Aa,oS,oC,1);
{
Dd&Bv=Ad[vi];vector<float>&Lz=Bs[vi];
int w=X1-X0+1,h=Y1-Y0+1;size_t sz=(size_t)w*h;
if(Jc[vi].size()<sz){Jc[vi].resize(sz);JR[vi].resize(sz);JU[vi].resize(sz);JS[vi].resize(sz);KM[vi].resize(sz);JT[vi].resize(sz);}
for(int yy=Y0;yy<=Y1;yy++){size_t sp=(size_t)yy*Al+X0,dp=(size_t)(yy-Y0)*w;
memcpy(&Jc[vi][dp],&Bv.nr[sp],w*sizeof(float));
memcpy(&JR[vi][dp],&Bv.ng[sp],w*sizeof(float));
memcpy(&JU[vi][dp],&Bv.nb[sp],w*sizeof(float));
memcpy(&JS[vi][dp],&Bv.dp[sp],w*sizeof(float));
memcpy(&KM[vi][dp],&Lz[sp],w*sizeof(float));
memcpy(&JT[vi][dp],&Bv.cov[sp],w*sizeof(uint8_t));}}
AF(vi,Al,X0,Y0,X1,Y1);
B0=1;
Bw(vi,Al,X0-Aa,Y0-Aa,X1+Aa,Y1+Aa,nS,nC,1);
HX[vi]=nS-oS;HY[vi]=nC-oC;}
double vs=B3[vi]+HX[vi],vc=(double)AT[vi]+(double)HY[vi];
ns+=vc>0?vs/vc:1.0;}
double Cd=ns/6.0;
if(Cd>=Ku){for(int vi=0;vi<6;vi++)if(bU[vi]){B3[vi]+=HX[vi];AT[vi]+=HY[vi];}
for(int f:vf[v])if(Ab[f])vf[u].push_back(f);
if(BL){if(FE[u].size()<FE[v].size())FE[u].swap(FE[v]);
FE[u].insert(FE[u].end(),FE[v].begin(),FE[v].end());FE[v].clear();
HE=max(HE,L6[0]);HD=max(HD,GV);++Lh;}
Ag[v]=0;uf[v]=u;--A3;
Cf[u]=L6[0];
++A4[u];++A4[v];
AM(Al,C9);
if(vf[u].size()>64){vector<int> Ci;Ci.reserve(32);
for(int f:vf[u])if(Ab[f])Ci.push_back(f);
vf[u].swap(Ci);}
Df=Cd;++AK;++IO;
Ai(u);
As(u,K6);
for(int w:K6)if(Ag[w])Ai(w);
for(int w:K6)if(Ag[w])IF(u,w);
if(IO%(D4?5000:32)==0){An(Al);Df=Ap();
if(D4){if(Df>=Ku-1e-12){if(!BL)FK();}
else{Eo();DR=true;return AK;}}
}}else{
X[u]=MK;Y[u]=MJ;Z[u]=MI;
for(auto&pr:FQ){int f=pr.first;if(pr.second==0)FA[f]=v;else if(pr.second==1)FB[f]=v;else FC[f]=v;}
Ab[G4[0]]=1;Ab[G4[1]]=1;
for(int vi=0;vi<6;vi++)if(bU[vi]){
Dd&Bv=Ad[vi];vector<float>&Lz=Bs[vi];
int X0=bB[vi][0],Y0=bB[vi][1],X1=bB[vi][2],Y1=bB[vi][3];int w=X1-X0+1;
for(int yy=Y0;yy<=Y1;yy++){size_t dp=(size_t)yy*Al+X0,sp=(size_t)(yy-Y0)*w;
memcpy(&Bv.nr[dp],&Jc[vi][sp],w*sizeof(float));
memcpy(&Bv.ng[dp],&JR[vi][sp],w*sizeof(float));
memcpy(&Bv.nb[dp],&JU[vi][sp],w*sizeof(float));
memcpy(&Bv.dp[dp],&JS[vi][sp],w*sizeof(float));
memcpy(&Lz[dp],&KM[vi][sp],w*sizeof(float));
memcpy(&Bv.cov[dp],&JT[vi][sp],w*sizeof(uint8_t));}
}
if(BL)++HU;
K2[bk]=make_pair(A4[u],A4[v]);}
continue;}
double Cd=-1e300;int bc=-1;
for(int ci=0;ci<Iu;ci++){X[u]=vp[ci][0];Y[u]=vp[ci][1];Z[u]=vp[ci][2];
double ns=0,ds[6];long dc[6];int rb[6][4];bool us[6];
for(int vi=0;vi<6;vi++){int c0,d0,c1,d1;
bool hb=AV(vi,C9,Al,c0,d0,c1,d1);
us[vi]=C6[vi]||hb;
if(!us[vi]){ds[vi]=0;dc[vi]=0;ns+=AT[vi]?(double)B3[vi]/AT[vi]:1.0;continue;}
int X0,Y0,X1,Y1;
if(C6[vi]&&hb){X0=min(ob[vi][0],c0);Y0=min(ob[vi][1],d0);X1=max(ob[vi][2],c1);Y1=max(ob[vi][3],d1);}else if(C6[vi]){X0=ob[vi][0];Y0=ob[vi][1];X1=ob[vi][2];Y1=ob[vi][3];}else{X0=c0;Y0=d0;X1=c1;Y1=d1;}
rb[vi][0]=X0;rb[vi][1]=Y0;rb[vi][2]=X1;rb[vi][3]=Y1;
Dp(vi,X0,Y0,X1,Y1,1,ds[vi],dc[vi]);
double vs=B3[vi]+ds[vi],vc=(double)AT[vi]+(double)dc[vi];
ns+=vc>0?vs/vc:1.0;}
ns/=6.0;
if(ns>Cd){Cd=ns;bc=ci;
for(int vi=0;vi<6;vi++){bU[vi]=us[vi];HX[vi]=ds[vi];HY[vi]=dc[vi];if(us[vi])for(int q2=0;q2<4;q2++)bB[vi][q2]=rb[vi][q2];}}
}
if(bc>=0&&Cd>=Ku){X[u]=vp[bc][0];Y[u]=vp[bc][1];Z[u]=vp[bc][2];
for(int vi=0;vi<6;vi++)if(bU[vi]){AF(vi,Al,bB[vi][0],bB[vi][1],bB[vi][2],bB[vi][3]);B3[vi]+=HX[vi];AT[vi]+=HY[vi];}
for(int f:vf[v])if(Ab[f])vf[u].push_back(f);
Ag[v]=0;uf[v]=u;--A3;
Cf[u]=L6[bc];
++A4[u];++A4[v];
AM(Al,C9);
if(vf[u].size()>64){vector<int> Ci;Ci.reserve(32);
for(int f:vf[u])if(Ab[f])Ci.push_back(f);
vf[u].swap(Ci);}
Df=Cd;++AK;
Ai(u);
As(u,K6);
for(int w:K6)if(Ag[w])Ai(w);
for(int w:K6)if(Ag[w])IF(u,w);
if(AK%32==0){An(Al);Df=Ap();}}else{
X[u]=MK;Y[u]=MJ;Z[u]=MI;
for(auto&pr:FQ){int f=pr.first;if(pr.second==0)FA[f]=v;else if(pr.second==1)FB[f]=v;else FC[f]=v;}
Ab[G4[0]]=1;Ab[G4[1]]=1;}}
return AK;};
auto Fl=[&](double GY)->int{
struct Tc{int a,b,c;double s;};
auto hk=[](int a,int b)->int64_t{if(a>b)swap(a,b);return((int64_t)a<<21)|b;};
unordered_map<int64_t,char> ue;ue.reserve((size_t)A3*8);
for(int f=0;f<NF;f++)if(Ab[f]){int q[3]={FA[f],FB[f],FC[f]};
for(int k=0;k<3;k++)ue[hk(q[k],q[(k+1)%3])]=1;}
vector<int> ds;for(int f=0;f<NF;f++)if(!Ab[f])ds.push_back(f);
vector<Tc> ca;vector<int> nb;
for(int v=0;v<NV;v++)if(Ag[v]){As(v,nb);
if(nb.size()<5||nb.size()>24)continue;
double tx=0,ty=0,tz=0;
for(int f:vf[v])if(Ab[f]&&(FA[f]==v||FB[f]==v||FC[f]==v)){int a=FA[f],b=FB[f],c=FC[f];
tx+=(Y[b]-Y[a])*(Z[c]-Z[a])-(Z[b]-Z[a])*(Y[c]-Y[a]);
ty+=(Z[b]-Z[a])*(X[c]-X[a])-(X[b]-X[a])*(Z[c]-Z[a]);
tz+=(X[b]-X[a])*(Y[c]-Y[a])-(Y[b]-Y[a])*(X[c]-X[a]);}
double tl=sqrt(tx*tx+ty*ty+tz*tz);if(tl<1e-15)continue;tx/=tl;ty/=tl;tz/=tl;
for(size_t i=0;i+2<nb.size();i++)for(size_t j=i+1;j+1<nb.size();j++)for(size_t k=j+1;k<nb.size();k++){int a=nb[i],b=nb[j],c=nb[k];
if(ue.count(hk(a,b))||ue.count(hk(b,c))||ue.count(hk(c,a)))continue;
double ux=X[b]-X[a],uy=Y[b]-Y[a],uz=Z[b]-Z[a],vx=X[c]-X[a],vy=Y[c]-Y[a],vz=Z[c]-Z[a],
nx=uy*vz-uz*vy,ny=uz*vx-ux*vz,nz=ux*vy-uy*vx,nl=sqrt(nx*nx+ny*ny+nz*nz);
if(nl<1e-15)continue;
if(nx*tx+ny*ty+nz*tz<0)swap(b,c),nx=-nx,ny=-ny,nz=-nz;
nx/=nl;ny/=nl;nz/=nl;
ca.push_back({a,b,c,max(0.0,1.0-(nx*tx+ny*ty+nz*tz))});}}
sort(ca.begin(),ca.end(),[](const Tc&a,const Tc&b){return a.s<b.s;});
double os=Df;int ac=0,di=0;vector<int> ff(2),Ci;
for(auto e:ca){if(ac>=F0||(double)(clock()-t0)>GY||di+1>=(int)ds.size())break;
int64_t ab=hk(e.a,e.b),bc=hk(e.b,e.c),cc=hk(e.c,e.a);
if(ue.count(ab)||ue.count(bc)||ue.count(cc))continue;
int f0=ds[di],f1=ds[di+1];ff[0]=f0;ff[1]=f1;
FA[f0]=FA[f1]=e.a;FB[f0]=FB[f1]=e.b;FC[f0]=FC[f1]=e.c;
Ab[f0]=Ab[f1]=1;AM(Al,ff);
double ns=0,dv[6];long dc[6];int bb[6][4];bool us[6];
for(int vi=0;vi<6;vi++){us[vi]=AV(vi,ff,Al,bb[vi][0],bb[vi][1],bb[vi][2],bb[vi][3]);
if(us[vi])Dp(vi,bb[vi][0],bb[vi][1],bb[vi][2],bb[vi][3],1,dv[vi],dc[vi]);else dv[vi]=0,dc[vi]=0;
double vs=B3[vi]+dv[vi],vc=(double)AT[vi]+dc[vi];ns+=vc>0?vs/vc:1.0;}
ns/=6;
if(ns>Df+1e-12){for(int vi=0;vi<6;vi++)if(us[vi]){AF(vi,Al,bb[vi][0],bb[vi][1],bb[vi][2],bb[vi][3]);B3[vi]+=dv[vi];AT[vi]+=dc[vi];}
Df=ns;ue[ab]=ue[bc]=ue[cc]=1;Ci.push_back(f0);Ci.push_back(f1);di+=2;++ac;}
else Ab[f0]=Ab[f1]=0,AM(Al,ff);}
for(int vi=0;vi<6;vi++){AP(X,Y,Z,FA,FB,FC,Ab,NF,Aw[vi][0],Aw[vi][1],Al,Ad[vi].nr,Ad[vi].ng,Ad[vi].nb,Ad[vi].dp,Ad[vi].cov);Bs[vi]=E5;}
An(Al);BC(Al);Df=Ap();
if(Df+1e-12<os){for(int f:Ci)Ab[f]=0;Ci.clear();
for(int vi=0;vi<6;vi++){AP(X,Y,Z,FA,FB,FC,Ab,NF,Aw[vi][0],Aw[vi][1],Al,Ad[vi].nr,Ad[vi].ng,Ad[vi].nb,Ad[vi].dp,Ad[vi].cov);Bs[vi]=E5;}
An(Al);BC(Al);Df=Ap();ac=0;}
Ee.assign(NF,0);A2.assign(NV,0);
for(int f:Ci){Ee[f]=1;Ab[f]=0;A2[FA[f]]=A2[FB[f]]=A2[FC[f]]=1;}
Bi();BC(Al);
return ac;};
auto FY=[&](double GY)->int{
int Fx=0;
vector<int64_t> IU;IU.reserve((size_t)A3*3+8);
for(int f=0;f<NF;f++)if(Ab[f]){int q3[3]={FA[f],FB[f],FC[f]};
if(q3[0]==q3[1]||q3[1]==q3[2]||q3[0]==q3[2])continue;
for(int k=0;k<3;k++){int a=q3[k],b=q3[(k+1)%3];if(a<b)IU.push_back(((int64_t)a<<32)|(uint32_t)b);}}
sort(IU.begin(),IU.end());IU.erase(unique(IU.begin(),IU.end()),IU.end());
vector<int> J2(NV,0);
for(int64_t k:IU){++J2[(int)(k>>32)];++J2[(int)(k&0xffffffff)];}
vector<int> Le(2);
for(int64_t k:IU){if((double)(clock()-t0)>GY)break;
int u=(int)(k>>32),v=(int)(k&0xffffffff);
if(!Ag[u]||!Ag[v])continue;
if(J2[u]<=3||J2[v]<=3)continue;
if(!A2.empty()&&(A2[u]||A2[v]))continue;
int G4[2],K5[2];
if(Ar(u,v,G4,K5)!=2)continue;
int f0=G4[0],f1=G4[1];
auto EG=[&](int f,int x,int y)->bool{return(FA[f]==x&&FB[f]==y)||(FB[f]==x&&FC[f]==y)||(FC[f]==x&&FA[f]==y);};
int A=u,B=v;
if(!EG(f0,A,B)){if(EG(f1,A,B))swap(f0,f1);else{A=v;B=u;
if(!EG(f0,A,B)){if(EG(f1,A,B))swap(f0,f1);else continue;}}}
auto Iq=[&](int f)->int{if(FA[f]!=A&&FA[f]!=B)return FA[f];if(FB[f]!=A&&FB[f]!=B)return FB[f];return FC[f];};
int c=Iq(f0),d=Iq(f1);
if(c==d||c==A||c==B||d==A||d==B)continue;
if(!A2.empty()&&(A2[c]||A2[d]))continue;
{bool ex=false;const vector<int>&GA=(vf[c].size()<=vf[d].size())?vf[c]:vf[d];
for(int f:GA)if(Ab[f]&&AE(f,c)&&AE(f,d)){ex=true;break;}
if(ex)continue;}
{double nx,ny,nz;
if(!Af(X[A],Y[A],Z[A],X[d],Y[d],Z[d],X[c],Y[c],Z[c],nx,ny,nz))continue;
if(!Af(X[d],Y[d],Z[d],X[B],Y[B],Z[B],X[c],Y[c],Z[c],nx,ny,nz))continue;}
Le[0]=f0;Le[1]=f1;
int ob[6][4];bool C6[6];
for(int vi=0;vi<6;vi++)C6[vi]=AV(vi,Le,Al,ob[vi][0],ob[vi][1],ob[vi][2],ob[vi][3]);
int N8=FA[f0],N6=FB[f0],N4=FC[f0],N7=FA[f1],N5=FB[f1],N3=FC[f1];
FA[f0]=A;FB[f0]=d;FC[f0]=c;
FA[f1]=d;FB[f1]=B;FC[f1]=c;
double ns=0,dv[6];long dc[6];int bb[6][4];bool us[6];bool Kv=true;
for(int vi=0;vi<6;vi++){int c0,d0,c1,d1;
bool hb=AV(vi,Le,Al,c0,d0,c1,d1);
us[vi]=C6[vi]||hb;
if(!us[vi]){dv[vi]=0;dc[vi]=0;ns+=AT[vi]?(double)B3[vi]/AT[vi]:1.0;continue;}
int X0,Y0,X1,Y1;
if(C6[vi]&&hb){X0=min(ob[vi][0],c0);Y0=min(ob[vi][1],d0);X1=max(ob[vi][2],c1);Y1=max(ob[vi][3],d1);}
else if(C6[vi]){X0=ob[vi][0];Y0=ob[vi][1];X1=ob[vi][2];Y1=ob[vi][3];}
else{X0=c0;Y0=d0;X1=c1;Y1=d1;}
if((long)(X1-X0+1)*(Y1-Y0+1)>40000){Kv=false;break;}
bb[vi][0]=X0;bb[vi][1]=Y0;bb[vi][2]=X1;bb[vi][3]=Y1;
Dp(vi,X0,Y0,X1,Y1,1,dv[vi],dc[vi]);
double vs=B3[vi]+dv[vi],vc=(double)AT[vi]+(double)dc[vi];
ns+=vc>0?vs/vc:1.0;}
if(Kv)ns/=6.0;
if(!Kv||ns<=Df+1e-9){FA[f0]=N8;FB[f0]=N6;FC[f0]=N4;FA[f1]=N7;FB[f1]=N5;FC[f1]=N3;continue;}
for(int vi=0;vi<6;vi++)if(us[vi]){AF(vi,Al,bb[vi][0],bb[vi][1],bb[vi][2],bb[vi][3]);B3[vi]+=dv[vi];AT[vi]+=dc[vi];}
AM(Al,Le);
auto El=[&](int w,int f){for(int g:vf[w])if(g==f)return;vf[w].push_back(f);};
El(d,f0);El(c,f1);
--J2[A];--J2[B];++J2[c];++J2[d];
++A4[A];++A4[B];++A4[c];++A4[d];
Df=ns;++Fx;
if(Fx%128==0){An(Al);Df=Ap();}}
return Fx;};
if(D4){int Dm=0;
while(!DR){double now=(double)(clock()-t0);
if(now>=D5||clock()>DE)break;
int rm=EJ(150,false);
FJ+=rm;
if(DR)break;
now=(double)(clock()-t0);
if(now>=D5||clock()>DE)break;
int mv=0;
if(rm<150){mv=JF(now+min(0.6*(double)CLOCKS_PER_SEC,0.30*(D5-now)));
H3+=mv;
if(mv==0&&rm==0){
now=(double)(clock()-t0);
if(Dm>=3||now>=D5||clock()>DE)break;
if(Dm==0)D5=min(D5,now+2.4*(double)CLOCKS_PER_SEC);
double Mt=(Dm==0?0.3:0.8)*(double)CLOCKS_PER_SEC;
int fl=FY(min(D5,now+Mt));
++Dm;Id+=fl;
if(D0)fprintf(stderr,"[flip] carve pass=%d fl=%d cur=%.6f t=%.2fs\n",Dm,fl,Df,(double)(clock()-t0)/CLOCKS_PER_SEC);
if(!fl)break;
An(Al);Df=Ap();}}}
An(Al);
double EI=Ap();
if(EI<BO-1e-9){Eo();
EI=Hu();
}}else{for(int rd=0;rd<6;rd++){double now=(double)(clock()-t0);
if(now>=D5||clock()>DE)break;
int mv=JF(now+0.45*(D5-now));
An(Al);Df=Ap();
int rm=EJ(1<<30,false);
An(Al);Df=Ap();
H3+=mv;FJ+=rm;
if(mv==0&&rm==0)break;}
if(NV>10000&&NV<=30000){
double IX=min(D5+2.4*(double)CLOCKS_PER_SEC,8.4*(double)CLOCKS_PER_SEC);
for(int xr=0;xr<3;xr++){double now=(double)(clock()-t0);
if(now>=IX||clock()>DE)break;
int fl=FY(min(IX,now+0.8*(double)CLOCKS_PER_SEC));
Id+=fl;
if(D0)fprintf(stderr,"[flip] tc3 xr=%d fl=%d cur=%.6f t=%.2fs\n",xr,fl,Df,(double)(clock()-t0)/CLOCKS_PER_SEC);
if(!fl)break;
An(Al);Df=Ap();
now=(double)(clock()-t0);
if(now>=IX||clock()>DE)break;
double Kk=D5;D5=IX;
int rm=EJ(1<<30,false);D5=Kk;
FJ+=rm;
An(Al);Df=Ap();
if(D0)fprintf(stderr,"[flip] tc3 xr=%d rm=%d cur=%.6f nAlive=%d t=%.2fs\n",xr,rm,Df,A3,(double)(clock()-t0)/CLOCKS_PER_SEC);
if(!rm)break;}}
if(NV>10000&&NV<=30000){double q=max(6.0*CLOCKS_PER_SEC,min(10.0*CLOCKS_PER_SEC,17.0*CLOCKS_PER_SEC-(double)t0));
int ac=Fl(q);
if(ac){D5=max(6.0*CLOCKS_PER_SEC,min(16.0*CLOCKS_PER_SEC,19.0*CLOCKS_PER_SEC-(double)t0));
int rm=EJ(1<<30,false);FJ+=rm;An(Al);Df=Ap();}}
double EI=Ap();
double KV=Em-1e-12;
if(CN>0.0&&H3+FJ>0)KV=Em+CN;
if(EI<KV){X=hX;Y=hY;Z=hZ;FA=Mo;FB=Mn;FC=Mm;Ab=Mg;Ag=Mf;vf=Mk;uf=Ml;A3=Hs;Cf=Ks;Ee.clear();A2.clear();H1=true;}}
if(NV<=10000&&A3>=5&&A3<=2000){
const int Hg=getenv("VR_PROF")?1:0;
if(H1){for(int vi=0;vi<6;vi++){AP(X,Y,Z,FA,FB,FC,Ab,NF,Aw[vi][0],Aw[vi][1],Al,Ad[vi].nr,Ad[vi].ng,Ad[vi].nb,Ad[vi].dp,Ad[vi].cov);Bs[vi]=E5;}
An(Al);BC(Al);}
Df=Ap();
double EE=Df;
double Cl=EE-1e-4;
if(EE>=BO-1e-9)Cl=BO;
else if(EE>=0.90&&Cl<0.90)Cl=0.90;
FK();
auto Ge=[&](double GY)->int{
int AK=0,HC=0;
vector<vector<int>> EM(NV);
for(int i=0;i<NV;i++){int r=H2(i);if(r>=0&&r<NV)EM[r].push_back(i);}
double I0=Aq*Aq;
double Ik=0.020*CZ;
vector<uint8_t> FI(NV,0);
vector<int> DL,Iy;
vector<pair<int,int>> J1;
vector<char> Gf;
auto H0=[&](int v,vector<int>&lp,vector<int>&fs)->bool{
lp.clear();fs.clear();
for(int f:vf[v]){if(f<0||f>=NF||!Ab[f])continue;
int a=FA[f],b=FB[f],c=FC[f];
if(a!=v&&b!=v&&c!=v)continue;
if(a==b||b==c||a==c)return false;
bool MS=false;
for(int g:fs)if(g==f){MS=true;break;}
if(!MS)fs.push_back(f);}
int k=(int)fs.size();
if(k<3)return false;
J1.clear();
for(int f:fs){int a=FA[f],b=FB[f],c=FC[f];
if(a==v)J1.push_back({b,c});else if(b==v)J1.push_back({c,a});else J1.push_back({a,b});}
Gf.assign(k,0);
Gf[0]=1;lp.push_back(J1[0].first);
int cw=J1[0].second;
for(int s=1;s<k;++s){lp.push_back(cw);
int nx=-1;
for(int j=0;j<k;++j)if(!Gf[j]&&J1[j].first==cw){nx=J1[j].second;Gf[j]=1;break;}
if(nx<0)return false;
cw=nx;}
if(cw!=lp[0]||(int)lp.size()!=k)return false;
for(int i=0;i<k;++i)for(int j=i+1;j<k;++j)if(lp[i]==lp[j])return false;
return true;};
auto E7=[&](int a,int b,const vector<int>&fs)->bool{
for(int f:vf[a]){if(f<0||f>=NF||!Ab[f])continue;
if(!AE(f,a)||!AE(f,b))continue;
bool in=false;
for(int g:fs)if(g==f){in=true;break;}
if(!in)return true;}
return false;};
auto Dg=[&](int p,int q)->int{int c2=0;
for(int f=0;f<NF;++f)if(Ab[f]){
int a=FA[f],b=FB[f],c3=FC[f];
if((a==p||b==p||c3==p)&&(a==q||b==q||c3==q))++c2;}
return c2;};
auto FH=[&](int v,const vector<int>&lp,const vector<int>&fs,int&Fv,double&Fu)->bool{
int k=(int)lp.size();
Fv=-1;Fu=1e300;
if(k<3||k>12)return false;
double FP=0,FO=0,FN=0,fn[12][3];
for(int t=0;t<k;++t){int f=fs[t];
double nx,ny,nz;
if(!Af(X[FA[f]],Y[FA[f]],Z[FA[f]],X[FB[f]],Y[FB[f]],Z[FB[f]],X[FC[f]],Y[FC[f]],Z[FC[f]],nx,ny,nz))return false;
fn[t][0]=nx;fn[t][1]=ny;fn[t][2]=nz;FP+=nx;FO+=ny;FN+=nz;}
double rl=sqrt(FP*FP+FO*FO+FN*FN);
if(rl<1e-12)return false;
FP/=rl;FO/=rl;FN/=rl;
double Dy=1.0;
for(int t=0;t<k;++t)Dy=min(Dy,fn[t][0]*FP+fn[t][1]*FO+fn[t][2]*FN);
if(Dy<0.70)return false;
double cx=0,cy=0,cz=0;
for(int x:lp){cx+=X[x];cy+=Y[x];cz+=Z[x];}
cx/=k;cy/=k;cz/=k;
double pd=fabs((X[v]-cx)*FP+(Y[v]-cy)*FO+(Z[v]-cz)*FN);
if(pd>Ik)return false;
for(int ai=0;ai<k;++ai){
bool M5=true;
for(int t=0;t<k-3&&M5;++t)if(E7(lp[ai],lp[(ai+2+t)%k],fs))M5=false;
if(!M5)continue;
double Er=1.0;
bool Kt=true;
for(int t=0;t<k-2;++t){int A=lp[ai],B=lp[(ai+1+t)%k],C=lp[(ai+2+t)%k];
double nx,ny,nz;
if(!Af(X[A],Y[A],Z[A],X[B],Y[B],Z[B],X[C],Y[C],Z[C],nx,ny,nz)){Kt=false;break;}
double d=nx*FP+ny*FO+nz*FN;
if(d<0.70){Kt=false;break;}
Er=min(Er,d);}
if(!Kt)continue;
bool MO=true;
for(int Kl:EM[v]){double bd=1e300;
for(int t=0;t<k-2;++t){int A=lp[ai],B=lp[(ai+1+t)%k],C=lp[(ai+2+t)%k];
double dd=CA(oX[Kl],oY[Kl],oZ[Kl],X[A],Y[A],Z[A],X[B],Y[B],Z[B],X[C],Y[C],Z[C]);
if(dd<bd)bd=dd;}
if(bd>I0){MO=false;break;}}
if(!MO)continue;
double cost=pd/CZ+0.6*(1.0-Dy)+0.4*(1.0-Er);
if(cost<Fu){Fu=cost;Fv=ai;}}
return Fv>=0;};
struct G3{int v;int apex;double cost;};
for(int round=0;round<8;++round){
if(AK>=64||HC>=400)break;
if((double)(clock()-t0)>GY||(double)clock()>DE)break;
vector<G3> FZ;
for(int v=0;v<NV;++v){
if(!Ag[v]||FI[v])continue;
if(!A2.empty()&&A2[v])continue;
if(!H0(v,DL,Iy))continue;
int ba;double Nh;
if(!FH(v,DL,Iy,ba,Nh))continue;
FZ.push_back({v,ba,Nh});}
if(Hg)fprintf(stderr,"[vremove] round=%d cands=%zu removed=%d evals=%d t=%.2fs\n",round,FZ.size(),AK,HC,(double)(clock()-t0)/CLOCKS_PER_SEC);
if(FZ.empty())break;
sort(FZ.begin(),FZ.end(),[](const G3&a,const G3&b){return a.cost<b.cost;});
int Hd=0;
for(const G3&cd:FZ){
if(AK>=64||HC>=400||A3<=4)break;
if((double)(clock()-t0)>GY||(double)clock()>DE)break;
int v=cd.v;
if(!Ag[v]||FI[v])continue;
if(!H0(v,DL,Iy))continue;
bool MN=true;
if(!A2.empty())for(int x:DL)if(A2[x]){MN=false;break;}
if(!MN){FI[v]=1;continue;}
int ai;double NI;
if(!FH(v,DL,Iy,ai,NI)){FI[v]=1;continue;}
int k=(int)DL.size(),nt=k-2;
int ob[6][4];bool C6[6];
for(int vi=0;vi<6;vi++)C6[vi]=AV(vi,Iy,Al,ob[vi][0],ob[vi][1],ob[vi][2],ob[vi][3]);
vector<array<int,5>> L7;L7.reserve(k);
for(int f:Iy)L7.push_back({f,FA[f],FB[f],FC[f],(int)Ab[f]});
for(int t=0;t<k;++t){int f=Iy[t];
if(t<nt){FA[f]=DL[ai];FB[f]=DL[(ai+1+t)%k];FC[f]=DL[(ai+2+t)%k];}
else Ab[f]=0;}
bool Ix=true;
for(int t=0;t<nt&&Ix;++t){int A=DL[ai],B=DL[(ai+1+t)%k],C=DL[(ai+2+t)%k];
if(Dg(A,B)!=2||Dg(B,C)!=2||Dg(A,C)!=2)Ix=false;}
if(!Ix){for(auto&r:L7){FA[r[0]]=r[1];FB[r[0]]=r[2];FC[r[0]]=r[3];Ab[r[0]]=(uint8_t)r[4];}
FI[v]=1;continue;}
AM(Al,Iy);
double ns=0,dv[6];long dc[6];int bb[6][4];bool us[6];
for(int vi=0;vi<6;vi++){int c0,d0,c1,d1;
bool hb=AV(vi,Iy,Al,c0,d0,c1,d1);
us[vi]=C6[vi]||hb;
if(!us[vi]){dv[vi]=0;dc[vi]=0;ns+=AT[vi]?(double)B3[vi]/AT[vi]:1.0;continue;}
int X0,Y0,X1,Y1;
if(C6[vi]&&hb){X0=min(ob[vi][0],c0);Y0=min(ob[vi][1],d0);X1=max(ob[vi][2],c1);Y1=max(ob[vi][3],d1);}
else if(C6[vi]){X0=ob[vi][0];Y0=ob[vi][1];X1=ob[vi][2];Y1=ob[vi][3];}
else{X0=c0;Y0=d0;X1=c1;Y1=d1;}
bb[vi][0]=X0;bb[vi][1]=Y0;bb[vi][2]=X1;bb[vi][3]=Y1;
Dp(vi,X0,Y0,X1,Y1,1,dv[vi],dc[vi]);
double vs=B3[vi]+dv[vi],vc=(double)AT[vi]+(double)dc[vi];
ns+=vc>0?vs/vc:1.0;}
ns/=6.0;++HC;
if(ns>=Cl){
for(int vi=0;vi<6;vi++)if(us[vi]){AF(vi,Al,bb[vi][0],bb[vi][1],bb[vi][2],bb[vi][3]);B3[vi]+=dv[vi];AT[vi]+=dc[vi];}
Ag[v]=0;--A3;
auto El=[&](int w,int f){for(int g:vf[w])if(g==f)return;vf[w].push_back(f);};
for(int t=0;t<nt;++t){int f=Iy[t];El(FA[f],f);El(FB[f],f);El(FC[f],f);}
int un=DL[0];double M2=1e300;
for(int x:DL){double dx=X[x]-X[v],dy=Y[x]-Y[v],dz=Z[x]-Z[v];
double d2=dx*dx+dy*dy+dz*dz;
if(d2<M2){M2=d2;un=x;}}
uf[v]=un;
Cf[un]=max(Cf[un],Cf[v]+sqrt(M2));Cf[v]=0.0;
if(un!=v){if(EM[un].size()<EM[v].size())EM[un].swap(EM[v]);
EM[un].insert(EM[un].end(),EM[v].begin(),EM[v].end());EM[v].clear();}
++A4[v];
for(int x:DL)++A4[x];
Df=ns;++AK;++Hd;
if(Hg)fprintf(stderr,"[vremove] commit v=%d k=%d ns=%.6f alive=%d\n",v,k,ns,A3);
}else{
for(auto&r:L7){FA[r[0]]=r[1];FB[r[0]]=r[2];FC[r[0]]=r[3];Ab[r[0]]=(uint8_t)r[4];}
AM(Al,Iy);
FI[v]=1;}}
if(!Hd)break;}
return AK;};
int vr=Ge((double)(clock()-t0)+1.5*(double)CLOCKS_PER_SEC);
if(vr){An(Al);Df=Ap();
if(Df<Cl-1e-12){Eo();vr=0;}}
if(Hg)fprintf(stderr,"[vremove] DONE removed=%d alive=%d startVr=%.6f cur=%.6f floor=%.6f t=%.2fs\n",vr,A3,EE,Df,Cl,(double)(clock()-t0)/CLOCKS_PER_SEC);
}
if(D0)fprintf(stderr,"[flip] done totFl=%d totRm=%d totMv=%d startS=%.6f nAlive=%d t=%.2fs\n",Id,FJ,H3,Em,A3,(double)(clock()-t0)/CLOCKS_PER_SEC);}
double Bc=0.25;
void Fq(){
if(NV <=10000){Ch=0.3;
BU=800.0;
Br=1000.0;
AL=1;
Bc=0.00310;
IQ=1;AW=0.55;
A7=0.87;
A1=0.895;
Dj=1;
G7=0.012;
BH=1;CB=0.050;BI=0.45;CU=0.965;Cs=0.030;
CX=1;Be=8;AI=0.992;AO=0.006;Bu=200.0;Bk=0;Bl=0;Au=0;A9=0.0460;
Cu=0;Bj=0;
HV=1;Bo=2.5;
BN=1;BX=1.0;AX=0.0013;DQ=0.13;}else if(NV <=30000){EL=0.5;
Ch=1.85;
B2=1;
BM=200.0;
Fb=1;
Ef=1.0;
E6=0.3;
Gt=0.65;
G7=0.042;
AL=1;
A0=1;
Ez=1;Hb=0.300;
Cj=1;CS=1;
Cw=1e-6;
B1=0.10;
Cg=0.10;AB=0.10;
Ed=1;
Cm=1;AN=0.28;Bm=3;
Bc=0.28483;
IQ=1;AW=0.75;
Cn=1;
BU=800;EN=1024;
A7=0.80;
A1=0.885;
BH=1;CB=0.00300;BI=0.27;CU=0.985;Cs=0.018;
DK=1;Ec=0.00500;
E4=1;H8=0.02;
Cu=1;Fh=0.72;
GQ=0.978;FW=0.024;
Bj=1;
CX=1;Be=11;
AI=0.985;AO=0.018;Bu=90.0;
Bk=1;Bl=1;Au=1;
Co=1;A9=0.0490;BS=0.35;
HV=1;Bo=5.985;HH=1;CN=1e-4;C3=7;AX=0.0015;DQ=0.13;}else if(NV <=45000){EL=6.0;
Ch=0.5;
BU=300.0;
B2=1;
Br=120.0;
G7=0.006;
AL=1;AJ=1;
A0=1;
Cj=1;CS=1;
Cw=1e-6;
B1=0.10;
Cg=0.10;AB=0.10;
Ed=1;
Ez=1;Hb=0.300;
Cm=1;AN=0.28;Bm=3;
Bc=0.14022;
MA=1;G6=3;CO=0.10;
IQ=1;AW=0.08;
Cn=1;BU=800;EN=1024;
BH=1;CB=0.00070;
BI=0.45;CU=0.965;Cs=0.030;
Cu=0;Bj=0;
A7=0.814;
A1=0.942;
CX=1;
Be=216;Fw=8;
AI=0.968;
AO=0.036;Bu=55.0;
HV=1;Bo=2.5;
BN=1;BX=1.1;AX=0.0015;}else if(NV <=70000){Ch=0.318;
Da=0.00987;
Bq=0.0506;
Br=107.28;
AL=1;
G7=0.0487;
A0=1;
Cj=0;
Cw=0.0;
B1=0.0;
Cg=0.0;
AB=0.0;
Ed=0;
Ct=1;
Cm=1;AN=0.402;Bm=3;
Bc=0.07640;
IQ=1;AW=0.55;Cn=1;BU=800;EN=1024;
BH=1;CB=0.00202;BI=0.40;CU=0.950;Cs=0.040;
Cu=0;Bj=0;
A7=0.502;
A1=0.933;
CX=1;Be=17;
AI=0.982;AO=0.024;
Bu=120.0;Bk=0;
Bl=0;Au=0;Co=0;
A9=0.0460;BS=0.45;
HV=1;Bo=4.0;
BN=1;BX=1.2;BO=0.9090;AX=0.0015;}else if(NV <=700000){EL=8.0;
Ch=0.35;
BU=800.0;
AL=1;
A0=1;
Cm=1;AN=0.40;Bm=4;
CX=1;Be=18;AI=0.982;AO=0.018;Bu=120.0;Bk=0;Bl=0;Au=0;A9=0.0460;
IQ=1;AW=0.45;
Bc=0.022163;
BH=0;CB=0.00100;
A7=0.85;
A1=0.947;HV=1;Bo=6.0;BN=1;BX=1.2;BO=0.9085;AX=0.0015;}else{
EL=3.0;
Ch=0.0;
BU=266.6666667;EN=256;
B2=0;
G7=0.0;
A0=1;
Ez=0;Hb=0.100;
Cm=0;
AN=0.28;
Bm=3;
CX=0;Be=19;AI=0.985;AO=0.020;Bu=120.0;Bk=0;Bl=0;Au=0;A9=0.0460;CK=13;Gm=3;
IQ=0;AW=0.05;
Bc=0.02400;
BH=0;CB=0.00010;
A7=0.809;
A1=0.95;
Cv=1;}
if(BN){DX=1;
EB=(NV>=250000?5:12);
Bc*=GR;}
else EB=32;
DM=(NV<=10000||(NV>30000&&NV<=70000));
}
int main(){HA=clock();Hl();
oX=X;oY=Y;oZ=Z;
Fq();
const long H4=-1;
const double Kq=1e300;
At=A7;
if(B1> 0.0||Cg> 0.0||AB> 0.0)
DS();
double Fg=1e300,Ff=1e300,Hp=1e300,F5=-1e300,F4=-1e300,H7=-1e300;
for(int i=0;i <NV;++i){Fg=min(Fg,X[i]);F5=max(F5,X[i]);
Ff=min(Ff,Y[i]);F4=max(F4,Y[i]);
Hp=min(Hp,Z[i]);H7=max(H7,Z[i]);}
double CZ=sqrt((F5-Fg)*(F5-Fg)+(F4-Ff)*(F4-Ff)+(H7-Hp)*(H7-Hp));
CV=CZ*CZ;
double Aq=0.045*CZ;
vector<double> Cf(NV,0.0);
long EP=H4>=0?H4:(long)floor(Bc*NV);
if(EP <4)EP=4;
vf.resize(NV);
Cp.assign(NV,0);
if(A0)Bd.assign(NF,1.0);
uf.resize(NV);for(int i=0;i <NV;++i)uf[i]=i;
Q.assign(NV,array<double,10>{});
for(auto&q:Q)q.fill(0.0);
for(int i=0;i <NV;++i){vf[i].clear();vf[i].reserve(16);}
Ab.assign(NF,1);
Ag.assign(NV,1);
A4.assign(NV,0);
for(int f=0;f <NF;++f){vf[FA[f]].push_back(f);vf[FB[f]].push_back(f);vf[FC[f]].push_back(f);}
auto KO=[&](int f)-> double{return H9==1?Eu[f]:H9==2?Fm[f]:Ae[f];};
double IM=1.0,HL=1.0;
if(Ch> 0.0||Da> 0.0||Br> 0.0||Bq> 0.0||BM> 0.0||Ez){BZ();
if(BM> 0.0)Dt();
double s=0;long n=0;
for(int f=0;f <NF;++f){double v=KO(f);s+=v;if(v> 0)++n;}
IM=(n> 0)?s/n:1.0;
if(Ez){double M3=0;long M4=0;
for(int f=0;f <NF;++f)if(D2[f]> 0){M3+=Ae[f];++M4;}
G2=(M4> 0&&M3> 0)?M3/M4:1.0;}
if(Br> 0.0||BM> 0.0){double ps=0;long pn=0;
for(int f=0;f <NF;++f)if(D2[f]> 0){ps+=Ae[f];++pn;}
D3=(pn> 0&&ps> 0)?ps/pn:1.0;}
if(Da> 0.0){BP.assign(NV,0.0);
for(int f=0;f <NF;++f){double c=Ae[f];BP[FA[f]]+=c;BP[FB[f]]+=c;BP[FC[f]]+=c;}
double cs=0;long n3=0;
for(int i=0;i <NV;++i){cs+=BP[i];if(BP[i]> 0)++n3;}
DH=(n3> 0&&cs> 0)?cs/n3:1.0;}
if(Dr> 0.0||AR> 0.0){Dc();
double cs=0;for(int f=0;f <NF;++f)cs+=Av[f];
HL=(NF> 0&&cs> 0)?cs/NF:1.0;}
if(Bq> 0.0){double cs=0;long n4=0;
for(int f=0;f <NF;++f){cs+=B7[f];if(B7[f]> 0)++n4;}
Go=(n4> 0&&cs> 0)?cs/n4:1.0;
AA.assign(NV,0.0);
for(int f=0;f <NF;++f)if(B7[f]> 0.0){double sv=B7[f]/Go;
if(sv> AA[FA[f]])AA[FA[f]]=sv;
if(sv> AA[FB[f]])AA[FB[f]]=sv;
if(sv> AA[FC[f]])AA[FC[f]]=sv;}}
}
vector<double> EC;
if(G7> 0.0){if((int)Av.size()!=NF)Dc();
vector<double> vc(NV,0.0),vk(NV,0.0);vector<int> vn(NV,0);
for(int f=0;f<NF;++f){double cp=D2[f]>0?Ae[f]:0.0;
double ck=Av[f];
vc[FA[f]]+=cp;vc[FB[f]]+=cp;vc[FC[f]]+=cp;
vk[FA[f]]+=ck;vk[FB[f]]+=ck;vk[FC[f]]+=ck;
vn[FA[f]]++;vn[FB[f]]++;vn[FC[f]]++;}
double Jy=0;long M0=0,Mz=0;double Jx=0;
for(int i=0;i<NV;++i){if(vc[i]>0){Jy+=vc[i];++M0;}if(vk[i]>0){Jx+=vk[i];++Mz;}}
Jy=(M0>0?Jy/M0:1.0);Jx=(Mz>0?Jx/Mz:1.0);
EC.assign(NV,1.0);
for(int v=0;v<NV;++v){double My=max(0.15,vc[v]/max(1e-9,Jy));
double Ln=vn[v]>0?vk[v]/(vn[v]*max(1e-9,Jx)):1.0;
EC[v]=sqrt(My*(1.0+min(1.2,G7*Ln)));}}
for(int f=0;f <NF;++f){int a=FA[f],b=FB[f],c=FC[f];
array<double,10> q=B6(a,b,c);
double Gz=1.0;
if(Ch> 0.0){double cov=pow(KO(f)/IM,K1);
double K0=1.0;
Gz+=Ch*cov*K0;}
if(Gz!=1.0)for(int k=0;k <10;++k)q[k]*=Gz;
double Lt=1.0;
if(Ez&&f <(int)Ae.size()){double cr=(G2> 0.0)?(Ae[f]/G2):1.0;
Lt=(1.0-Hb)+Hb*cr;
if(Lt <0.0)Lt=0.0;
for(int k=0;k <10;++k)q[k]*=Lt;}
if(A0&&(int)Bd.size()==NF)Bd[f]=Gz*Lt;
Ak(Q[a],q);Ak(Q[b],q);Ak(Q[c],q);}
A3=NV;
priority_queue<AH> pq;
auto DN=[&](int u,int v){double LL=A1;
if(Da> 0.0){double cn=0.5*(BP[u]+BP[v])/DH;
LL=A1*(1.0+Da*(cn-1.0));
if(LL <0.02)LL=0.02;}
return LL;};
auto BA=[&](int u,int v,const int fs[2],
const array<double,10>&q,
double px,double py,double pz,
double Jf){double cost=max(0.0,Jf);
if(Cw> 0.0){double dx=X[u]-X[v],dy=Y[u]-Y[v],dz=Z[u]-Z[v];
double Li=dx*dx+dy*dy+dz*dz;
double KK=max(0.0,q[0]+q[4]+q[7]);
cost=max(cost,Cw*KK*Li);}
double Im=0.0;
if(Br> 0){double t=DF(u,v,fs[0],fs[1],px,py,pz);
Im+=Br*t;cost*=1.0+Br*t;}
if(BM> 0){double t=DP(u,v,fs[0],fs[1],px,py,pz);
Im+=BM*t;cost*=1.0+BM*t;}
if(Bq> 0.0){double sp=Bq*max(AA[u],AA[v]);
Im+=sp;cost*=1.0+sp;}
if(Fc> 0.0&&Im> 0.0)
cost=max(cost,Fc*CV*Im);
if(G7> 0.0)cost*=sqrt(EC[u]*EC[v]);
return cost;};
auto EZ=[&](int u,int v,const int fs[2],
const array<double,10>&q,
double&px,double&py,double&pz,
double&D8)-> bool{
Am(u,v,q,px,py,pz,D8);
if(!(CS&&B2))return true;
double qx=px,qy=py,qz=pz;
double mx=0.5*(X[u]+X[v]),my=0.5*(Y[u]+Y[v]),mz=0.5*(Z[u]+Z[v]);
double Ac[20][3];int Ah=4;
Ac[0][0]=qx;Ac[0][1]=qy;Ac[0][2]=qz;
Ac[1][0]=mx;Ac[1][1]=my;Ac[1][2]=mz;
Ac[2][0]=X[u];Ac[2][1]=Y[u];Ac[2][2]=Z[u];
Ac[3][0]=X[v];Ac[3][1]=Y[v];Ac[3][2]=Z[v];
const double ts[4]={0.2,0.4,0.6,0.8};
for(double t:ts){Ac[Ah][0]=qx+t*(mx-qx);Ac[Ah][1]=qy+t*(my-qy);Ac[Ah][2]=qz+t*(mz-qz);++Ah;}
for(double t:ts){Ac[Ah][0]=qx+t*(X[u]-qx);Ac[Ah][1]=qy+t*(Y[u]-qy);Ac[Ah][2]=qz+t*(Z[u]-qz);++Ah;}
for(double t:ts){Ac[Ah][0]=qx+t*(X[v]-qx);Ac[Ah][1]=qy+t*(Y[v]-qy);Ac[Ah][2]=qz+t*(Z[v]-qz);++Ah;}
double DZ=At;
At=A7;
double CC=DN(u,v);
bool Do=!A5&&
B1<=0.0&&AB<=0.0&&
1;
if(Do)B9(u,v,fs[0],fs[1]);
else B8(u,v,fs[0],fs[1]);
bool ok=false;double AU=1e300;
double bx=qx,by=qy,bz=qz;
for(int i=0;i<Ah;++i){double cx=Ac[i][0],cy=Ac[i][1],cz=Ac[i][2];
double du=sqrt((cx-X[u])*(cx-X[u])+(cy-Y[u])*(cy-Y[u])+(cz-Z[u])*(cz-Z[u]));
double dv=sqrt((cx-X[v])*(cx-X[v])+(cy-Y[v])*(cy-Y[v])+(cz-Z[v])*(cz-Z[v]));
double dn=max(Cf[u]+du,Cf[v]+dv)*CC;
if(dn>Aq)continue;
double rc;
if(Do){rc=BW(u,v,cx,cy,cz);
if(rc>=1e300)continue;
if(IQ)rc+=AW*DA(u,v,fs[0],fs[1],cx,cy,cz);}else{
double Iv;
if(!BV(u,v,cx,cy,cz,rc,Iv))continue;
if(IQ)rc+=AW*Iv;}
if(rc<AU){AU=rc;bx=cx;by=cy;bz=cz;ok=true;}}
At=DZ;
if(!ok)return false;
px=bx;py=by;pz=bz;
D8=max(0.0,AG(q,px,py,pz));
return true;};
auto Cq=[&](int a,int b){if(a==b)return;
if(a>b)swap(a,b);
if(!Ag[a]||!Ag[b])return;
int fs[2],Gu[2];
if(Ar(a,b,fs,Gu)!=2)return;
array<double,10> q=Q[a];Ak(q,Q[b]);
double px,py,pz,MF;
if(!EZ(a,b,fs,q,px,py,pz,MF))return;
double cost=BA(a,b,fs,q,px,py,pz,MF);
if(!isfinite(cost))return;
pq.push({cost,a,b,A4[a],A4[b]});};
for(int f=0;f <NF;++f){int Hz[3]={FA[f],FB[f],FC[f]};
for(int e=0;e <3;++e){int u=Hz[e],v=Hz[(e+1)%3];
if(u>v)continue;
Cq(u,v);}}
vector<int> nb;
nb.reserve(32);
vector<int> Fn;
vector<int> Bf,CL;
vector<pair<int,int>> Aj;
Bf.reserve(64);CL.reserve(32);Aj.reserve(256);
while(A3> EP&&!pq.empty()){AH ed=pq.top();pq.pop();
int u=ed.u,v=ed.v;
if(!Ag[u]||!Ag[v]||ed.ver_u!=A4[u]||ed.ver_v!=A4[v])continue;
int fs[2],Gu[2];
int sc=Ar(u,v,fs,Gu);
if(sc!=2)continue;
array<double,10> q=Q[u];Ak(q,Q[v]);
double px,py,pz,cost;
Am(u,v,q,px,py,pz,cost);
if(cost> Kq)continue;
if(!AZ(u,v,Gu))continue;
At=A7;
double CC=A1;
if(Da> 0.0){double cn=0.5*(BP[u]+BP[v])/DH;
CC=A1*(1.0+Da*(cn-1.0));
if(CC <0.02)CC=0.02;}
double Dq;
{
double mx=0.5*(X[u]+X[v]),my=0.5*(Y[u]+Y[v]),mz=0.5*(Z[u]+Z[v]);
double Ac[20][3];int Ah=1;
Ac[0][0]=px;Ac[0][1]=py;Ac[0][2]=pz;
if(B2){Ac[1][0]=mx;Ac[1][1]=my;Ac[1][2]=mz;
Ac[2][0]=X[u];Ac[2][1]=Y[u];Ac[2][2]=Z[u];
Ac[3][0]=X[v];Ac[3][1]=Y[v];Ac[3][2]=Z[v];Ah=4;
const double ts[4]={0.2,0.4,0.6,0.8};
for(double t:ts){Ac[Ah][0]=px+t*(mx-px);Ac[Ah][1]=py+t*(my-py);Ac[Ah][2]=pz+t*(mz-pz);++Ah;}
for(double t:ts){Ac[Ah][0]=px+t*(X[u]-px);Ac[Ah][1]=py+t*(Y[u]-py);Ac[Ah][2]=pz+t*(Z[u]-pz);++Ah;}
for(double t:ts){Ac[Ah][0]=px+t*(X[v]-px);Ac[Ah][1]=py+t*(Y[v]-py);Ac[Ah][2]=pz+t*(Z[v]-pz);++Ah;}}else if(Dj==2){const double ts[5]={0.15,0.30,0.50,0.70,1.0};
for(double t:ts){Ac[Ah][0]=px+t*(mx-px);Ac[Ah][1]=py+t*(my-py);Ac[Ah][2]=pz+t*(mz-pz);++Ah;}
Ac[Ah][0]=X[u];Ac[Ah][1]=Y[u];Ac[Ah][2]=Z[u];++Ah;
Ac[Ah][0]=X[v];Ac[Ah][1]=Y[v];Ac[Ah][2]=Z[v];++Ah;}else if(Dj==1){Ac[1][0]=mx;Ac[1][1]=my;Ac[1][2]=mz;
Ac[2][0]=X[u];Ac[2][1]=Y[u];Ac[2][2]=Z[u];
Ac[3][0]=X[v];Ac[3][1]=Y[v];Ac[3][2]=Z[v];Ah=4;}
bool ok=false;
if(B2){double AU=1e300;
bool Do=!A5&&
B1 <=0.0&&AB <=0.0&&
1;
if(Do)B9(u,v,fs[0],fs[1]);
else B8(u,v,fs[0],fs[1]);
for(int c=0;c <Ah;++c){double cx=Ac[c][0],cy=Ac[c][1],cz=Ac[c][2];
double du=sqrt((cx-X[u])*(cx-X[u])+(cy-Y[u])*(cy-Y[u])+(cz-Z[u])*(cz-Z[u]));
double Jv=sqrt((cx-X[v])*(cx-X[v])+(cy-Y[v])*(cy-Y[v])+(cz-Z[v])*(cz-Z[v]));
double dn=max(Cf[u]+du,Cf[v]+Jv)*CC;
if(dn> Aq)continue;
double rc;
if(Do){rc=BW(u,v,cx,cy,cz);
if(rc>=1e300)continue;
if(IQ)rc+=AW*DA(u,v,fs[0],fs[1],cx,cy,cz);}else{
double Iv;
if(!BV(u,v,cx,cy,cz,rc,Iv))continue;
if(IQ)rc+=AW*Iv;}
if(rc <AU){AU=rc;px=cx;py=cy;pz=cz;Dq=dn;ok=true;}}
}else{for(int c=0;c <Ah;++c){double cx=Ac[c][0],cy=Ac[c][1],cz=Ac[c][2];
double du=sqrt((cx-X[u])*(cx-X[u])+(cy-Y[u])*(cy-Y[u])+(cz-Z[u])*(cz-Z[u]));
double Jv=sqrt((cx-X[v])*(cx-X[v])+(cy-Y[v])*(cy-Y[v])+(cz-Z[v])*(cz-Z[v]));
double dn=max(Cf[u]+du,Cf[v]+Jv)*CC;
if(dn> Aq)continue;
if(!BJ(u,v,fs[0],fs[1],cx,cy,cz))continue;
px=cx;py=cy;pz=cz;Dq=dn;ok=true;break;}}
if(!ok)continue;}
if(A0&&Cv){Fn.clear();
for(int f:vf[u])if(Ab[f])Fn.push_back(f);
for(int f:vf[v])if(Ab[f])Fn.push_back(f);
sort(Fn.begin(),Fn.end());Fn.erase(unique(Fn.begin(),Fn.end()),Fn.end());
for(int f:Fn){int a=FA[f],b=FB[f],c=FC[f];
array<double,10> fq=B6(a,b,c);
double m=f<(int)Bd.size()?Bd[f]:1.0;
int vs[3]={a,b,c};
for(int w:vs)for(int k=0;k<10;k++)Q[w][k]-=fq[k]*m;}}
X[u]=px;Y[u]=py;Z[u]=pz;
Cf[u]=Dq;
if(!A0)Ak(Q[u],Q[v]);
Ab[fs[0]]=0;Ab[fs[1]]=0;
for(int f:vf[v]){if(!Ab[f])continue;
if(FA[f]==v)FA[f]=u;
if(FB[f]==v)FB[f]=u;
if(FC[f]==v)FC[f]=u;
vf[u].push_back(f);}
Ag[v]=0;
uf[v]=u;
--A3;
if(A0&&Cv)for(int f:Fn)if(Ab[f]){int a=FA[f],b=FB[f],c=FC[f];
array<double,10> fq=B6(a,b,c);
double m=f<(int)Bd.size()?Bd[f]:1.0;
int vs[3]={a,b,c};
for(int w:vs)for(int k=0;k<10;k++)Q[w][k]+=fq[k]*m;}
if(vf[u].size()> 64){vector<int> Ci;
Ci.reserve(32);
for(int f:vf[u])if(Ab[f])Ci.push_back(f);
vf[u].swap(Ci);}
As(u,nb);
if(Cj){Bf.clear();
Bf.push_back(u);
for(int w:nb)if(Ag[w])Bf.push_back(w);
if(A0&&!Cv)
for(int x:Bf)Ai(x);
++A4[v];
for(int x:Bf)++A4[x];
Aj.clear();
for(int x:Bf){As(x,CL);
for(int y:CL){if(!Ag[y]||x==y)continue;
Aj.emplace_back(min(x,y),max(x,y));}}
sort(Aj.begin(),Aj.end());
Aj.erase(unique(Aj.begin(),Aj.end()),Aj.end());
for(const auto&e:Aj)Cq(e.first,e.second);}else{
++A4[u];
++A4[v];
if(A0&&!Cv){Ai(u);
for(int w:nb)Ai(w);}
for(int w:nb)Cq(u,w);}}
if(BH&&CB> 0.0){int BK=(int)floor(CB*NV);
if(BK <1&&NV> 1000)BK=1;
if(Dl)
BK=min(BK,max(0,A3-(int)EP));
if(BK> 0){if(Ed||(int)D2.size()!=NF){BZ();
En();}
vector<double> CT(NV,0.0);
for(int f=0;f <NF;++f)if(Ab[f]){int a=FA[f],b=FB[f],c=FC[f];
if(a==b||b==c||a==c)continue;
double cv=(f <(int)Ae.size())?Ae[f]:0.0;
CT[a]+=cv;CT[b]+=cv;CT[c]+=cv;}
double Fi=0.0;long Hr=0;
for(int i=0;i <NV;++i)if(Ag[i]){Fi+=CT[i];++Hr;}
Fi=(Hr> 0&&Fi> 0.0)?(Fi/Hr):1.0;
double Ek=BI;
double CH=CU;
double C4=Cs;
double Du=0.15;
int BT=0;
auto BR=[&](int x)->bool{if(!Ag[x])return false;
double cn=CT[x]/Fi;
if(cn> Ek)return false;
if((int)AA.size()==NV&&AA[x]> Du)return false;
return true;};
int EQ=max(0,(int)floor(Ec*NV));
int CY=0;
bool Dv=DK&&EQ==0;
priority_queue<AH> K9;
auto Ce=[&](int a,int b,
double&bx,double&by,double&bz,
double&M1,double&Ls)->bool{
int IT[2],Kg[2];
if(Ar(a,b,IT,Kg)!=2)return false;
array<double,10> qq=Q[a];Ak(qq,Q[b]);
double qx,qy,qz,qc;
Am(a,b,qq,qx,qy,qz,qc);
double mx=0.5*(X[a]+X[b]),my=0.5*(Y[a]+Y[b]),mz=0.5*(Z[a]+Z[b]);
double Ac[20][3];int Ah=4;
Ac[0][0]=qx;Ac[0][1]=qy;Ac[0][2]=qz;
Ac[1][0]=mx;Ac[1][1]=my;Ac[1][2]=mz;
Ac[2][0]=X[a];Ac[2][1]=Y[a];Ac[2][2]=Z[a];
Ac[3][0]=X[b];Ac[3][1]=Y[b];Ac[3][2]=Z[b];
if(Bj){const double ts[4]={0.2,0.4,0.6,0.8};
for(double t:ts){Ac[Ah][0]=qx+t*(mx-qx);Ac[Ah][1]=qy+t*(my-qy);Ac[Ah][2]=qz+t*(mz-qz);++Ah;}
for(double t:ts){Ac[Ah][0]=qx+t*(X[a]-qx);Ac[Ah][1]=qy+t*(Y[a]-qy);Ac[Ah][2]=qz+t*(Z[a]-qz);++Ah;}
for(double t:ts){Ac[Ah][0]=qx+t*(X[b]-qx);Ac[Ah][1]=qy+t*(Y[b]-qy);Ac[Ah][2]=qz+t*(Z[b]-qz);++Ah;}}
double DZ=At;
At=CH;
bool ok=false;
double AU=1e300;
for(int ci=0;ci<Ah;++ci){double cx=Ac[ci][0],cy=Ac[ci][1],cz=Ac[ci][2];
if(!BJ(a,b,IT[0],IT[1],cx,cy,cz))continue;
double nc=Bh(a,b,IT[0],IT[1],cx,cy,cz);
if(nc>C4)continue;
double du=sqrt((cx-X[a])*(cx-X[a])+(cy-Y[a])*(cy-Y[a])+(cz-Z[a])*(cz-Z[a]));
double Jv=sqrt((cx-X[b])*(cx-X[b])+(cy-Y[b])*(cy-Y[b])+(cz-Z[b])*(cz-Z[b]));
double dn=max(Cf[a]+du,Cf[b]+Jv)*A1;
if(dn>Aq*0.98)continue;
double qv=max(0.0,AG(qq,cx,cy,cz));
double rp=BE(a,b,IT[0],IT[1],cx,cy,cz);
double Hk=rp;
double sc;
if(E4==2){sc=BA(a,b,IT,qq,cx,cy,cz,qv);
}else{
sc=Hk+H8*qv/max(1e-18,CV)
+0.10*nc;}
if(sc<AU){AU=sc;bx=cx;by=cy;bz=cz;M1=dn;Ls=sc;ok=true;}}
At=DZ;
return ok;};
auto BQ=[&](int a,int b){if(a==b)return;
if(a>b)swap(a,b);
if(!BR(a)||!BR(b))return;
int IT[2],Kg[2];
if(Ar(a,b,IT,Kg)!=2)return;
double GJ;
if(Dv){double bx,by,bz,bd;
if(!Ce(a,b,bx,by,bz,bd,GJ))return;}else{
array<double,10> qq=Q[a];Ak(qq,Q[b]);
double qx,qy,qz;
Am(a,b,qq,qx,qy,qz,GJ);}
if(!isfinite(GJ))return;
K9.push({GJ,a,b,A4[a],A4[b]});};
auto B4=[&](){priority_queue<AH> empty;
K9.swap(empty);
for(int f=0;f <NF;++f)if(Ab[f]){int Hz[3]={FA[f],FB[f],FC[f]};
for(int e=0;e <3;++e){int a=Hz[e],b=Hz[(e+1)%3];
if(a> b)continue;
BQ(a,b);}}
};
B4();
vector<int> La;La.reserve(32);
while(CY <BK){if(DK&&!Dv&&CY>=EQ){Dv=true;
B4();
continue;}
if(K9.empty()){if(!Cu||BT>=Es)break;
++BT;
double t=(double)BT/(double)max(1,Es);
Ek=BI+t*(Fh-BI);
CH=CU+t*(GQ-CU);
C4=Cs+t*(FW-Cs);
Du=0.15+t*(Jk-0.15);
B4();
if(K9.empty()&&BT>=Es)break;
continue;}
AH ed=K9.top();K9.pop();
int u=ed.u,v=ed.v;
if(!Ag[u]||!Ag[v]||ed.ver_u!=A4[u]||ed.ver_v!=A4[v])continue;
if(!BR(u)||!BR(v))continue;
int fs[2],Gu[2];
if(Ar(u,v,fs,Gu)!=2)continue;
if(!AZ(u,v,Gu))continue;
array<double,10> q=Q[u];Ak(q,Q[v]);
double HK=0.0,HJ=0.0,HI=0.0,Dq=0.0;
bool ok=false;
if(Dv){double GJ=0.0;
ok=Ce(u,v,HK,HJ,HI,Dq,GJ);}else{
double px,py,pz,cost;
Am(u,v,q,px,py,pz,cost);
double mx=0.5*(X[u]+X[v]),my=0.5*(Y[u]+Y[v]),mz=0.5*(Z[u]+Z[v]);
double Ac[20][3];int Ah=4;
Ac[0][0]=px;Ac[0][1]=py;Ac[0][2]=pz;
Ac[1][0]=mx;Ac[1][1]=my;Ac[1][2]=mz;
Ac[2][0]=X[u];Ac[2][1]=Y[u];Ac[2][2]=Z[u];
Ac[3][0]=X[v];Ac[3][1]=Y[v];Ac[3][2]=Z[v];
if(Bj){const double ts[4]={0.2,0.4,0.6,0.8};
for(double t:ts){Ac[Ah][0]=px+t*(mx-px);Ac[Ah][1]=py+t*(my-py);Ac[Ah][2]=pz+t*(mz-pz);++Ah;}
for(double t:ts){Ac[Ah][0]=px+t*(X[u]-px);Ac[Ah][1]=py+t*(Y[u]-py);Ac[Ah][2]=pz+t*(Z[u]-pz);++Ah;}
for(double t:ts){Ac[Ah][0]=px+t*(X[v]-px);Ac[Ah][1]=py+t*(Y[v]-py);Ac[Ah][2]=pz+t*(Z[v]-pz);++Ah;}}
double DZ=At;
At=CH;
double Ga=1e300;
HK=px;HJ=py;HI=pz;
for(int ci=0;ci <Ah;++ci){double cx=Ac[ci][0],cy=Ac[ci][1],cz=Ac[ci][2];
if(!BJ(u,v,fs[0],fs[1],cx,cy,cz))continue;
double nc=Bh(u,v,fs[0],fs[1],cx,cy,cz);
if(nc> C4)continue;
double du=sqrt((cx-X[u])*(cx-X[u])+(cy-Y[u])*(cy-Y[u])+(cz-Z[u])*(cz-Z[u]));
double Jv=sqrt((cx-X[v])*(cx-X[v])+(cy-Y[v])*(cy-Y[v])+(cz-Z[v])*(cz-Z[v]));
double dn=max(Cf[u]+du,Cf[v]+Jv)*A1;
if(dn> Aq*0.98)continue;
double sc=AG(q,cx,cy,cz)+CV*nc;
if(sc <Ga){Ga=sc;HK=cx;HJ=cy;HI=cz;Dq=dn;ok=true;}}
At=DZ;}
if(!ok)continue;
X[u]=HK;Y[u]=HJ;Z[u]=HI;
Cf[u]=Dq;
if(Ct||!A0)Ak(Q[u],Q[v]);
Ab[fs[0]]=0;Ab[fs[1]]=0;
for(int f:vf[v]){if(!Ab[f])continue;
if(FA[f]==v)FA[f]=u;
if(FB[f]==v)FB[f]=u;
if(FC[f]==v)FC[f]=u;
vf[u].push_back(f);}
Ag[v]=0;
uf[v]=u;
--A3;
++CY;
if(vf[u].size()> 64){vector<int> Ci;
Ci.reserve(32);
for(int f:vf[u])if(Ab[f])Ci.push_back(f);
vf[u].swap(Ci);}
As(u,La);
if(Ct){++A4[u];
++A4[v];
for(int w:La)BQ(u,w);}else if(A0){Bf.clear();Bf.push_back(u);
for(int w:La)if(Ag[w])Bf.push_back(w);
for(int x:Bf)Ai(x);
++A4[v];
for(int x:Bf)++A4[x];
for(int x:Bf){double cv=0.0;
for(int f:vf[x])if(Ab[f]&&
(FA[f]==x||FB[f]==x||FC[f]==x))
cv+=(f<(int)Ae.size()?Ae[f]:0.0);
CT[x]=cv;}
CT[v]=0.0;
Aj.clear();
for(int x:Bf){As(x,CL);
for(int y:CL)if(Ag[y]&&x!=y)
Aj.emplace_back(min(x,y),max(x,y));}
sort(Aj.begin(),Aj.end());
Aj.erase(unique(Aj.begin(),Aj.end()),Aj.end());
for(const auto&e:Aj)BQ(e.first,e.second);}else{
++A4[u];
++A4[v];
for(int w:La)BQ(u,w);}}
}}
Fe(Aq,Cf);
if(MA)Hh(Aq,Cf);
Di(CZ,Cf,EP);
G5(CZ,Cf,Aq);
Hy(CZ,Cf,Aq);if(NV>700000){int _fe=0;for(int _p=0;_p<3;++_p){for(int v=0;v<NV;++v)if(Ag[v]){vector<int>_nb;As(v,_nb);for(int u:_nb)if(Ag[u]&&u>v){int fs[2],op[2];if(Ar(u,v,fs,op)!=2)continue;double _px=(X[u]+X[v])*.5,_py=(Y[u]+Y[v])*.5,_pz=(Z[u]+Z[v])*.5;if(!AZ(u,v,op))continue;bool _ok=1;for(int s=0;s<2&&_ok;++s){int x=s?u:v;for(int f:vf[x])if(Ab[f]){int a=FA[f],b=FB[f],c=FC[f];if(a==b||b==c||a==c)continue;double on[3];if(!Af(X[a],Y[a],Z[a],X[b],Y[b],Z[b],X[c],Y[c],Z[c],on[0],on[1],on[2]))continue;auto _gx=[&](int w){return(w==x)?_px:X[w];};auto _gy=[&](int w){return(w==x)?_py:Y[w];};auto _gz=[&](int w){return(w==x)?_pz:Z[w];};double nn[3];if(!Af(_gx(a),_gy(a),_gz(a),_gx(b),_gy(b),_gz(b),_gx(c),_gy(c),_gz(c),nn[0],nn[1],nn[2])){_ok=0;break;}double nd=Ba(on[0],on[1],on[2],nn[0],nn[1],nn[2]);if(nd>0.00035){_ok=0;break;}}}if(!_ok)continue;X[v]=_px;Y[v]=_py;Z[v]=_pz;Ab[fs[0]]=Ab[fs[1]]=0;for(int f:vf[u])if(Ab[f]){if(FA[f]==u)FA[f]=v;if(FB[f]==u)FB[f]=v;if(FC[f]==u)FC[f]=v;vf[v].push_back(f);}Ag[u]=0;uf[u]=v;--A3;++_fe;if(vf[v].size()>64){vector<int>_cc;_cc.reserve(32);for(int f:vf[v])if(Ab[f])_cc.push_back(f);vf[v].swap(_cc);}break;}}if(_fe==0)goto _ed7;}_ed7:;}if(NV>10000&&NV<=30000){int _fe=0;for(int _p=0;_p<3;++_p){for(int v=0;v<NV;++v)if(Ag[v]){vector<int>_nb;As(v,_nb);for(int u:_nb)if(Ag[u]&&u>v){int fs[2],op[2];if(Ar(u,v,fs,op)!=2)continue;double _px=(X[u]+X[v])*.5,_py=(Y[u]+Y[v])*.5,_pz=(Z[u]+Z[v])*.5;if(!AZ(u,v,op))continue;bool _ok=1;for(int s=0;s<2&&_ok;++s){int x=s?u:v;for(int f:vf[x])if(Ab[f]){int a=FA[f],b=FB[f],c=FC[f];if(a==b||b==c||a==c)continue;double on[3];if(!Af(X[a],Y[a],Z[a],X[b],Y[b],Z[b],X[c],Y[c],Z[c],on[0],on[1],on[2]))continue;auto _gx=[&](int w){return(w==x)?_px:X[w];};auto _gy=[&](int w){return(w==x)?_py:Y[w];};auto _gz=[&](int w){return(w==x)?_pz:Z[w];};double nn[3];if(!Af(_gx(a),_gy(a),_gz(a),_gx(b),_gy(b),_gz(b),_gx(c),_gy(c),_gz(c),nn[0],nn[1],nn[2])){_ok=0;break;}double nd=Ba(on[0],on[1],on[2],nn[0],nn[1],nn[2]);if(nd>0.00070){_ok=0;break;}}}if(!_ok)continue;X[v]=_px;Y[v]=_py;Z[v]=_pz;Ab[fs[0]]=Ab[fs[1]]=0;for(int f:vf[u])if(Ab[f]){if(FA[f]==u)FA[f]=v;if(FB[f]==u)FB[f]=v;if(FC[f]==u)FC[f]=v;vf[v].push_back(f);}Ag[u]=0;uf[u]=v;--A3;++_fe;if(vf[v].size()>64){vector<int>_cc;_cc.reserve(32);for(int f:vf[v])if(Ab[f])_cc.push_back(f);vf[v].swap(_cc);}break;}}if(_fe==0)goto _ed3;}_ed3:;}if(NV>30000&&NV<=45000){int _fe=0;for(int _p=0;_p<3;++_p){for(int v=0;v<NV;++v)if(Ag[v]){vector<int>_nb;As(v,_nb);for(int u:_nb)if(Ag[u]&&u>v){int fs[2],op[2];if(Ar(u,v,fs,op)!=2)continue;double _px=(X[u]+X[v])*.5,_py=(Y[u]+Y[v])*.5,_pz=(Z[u]+Z[v])*.5;if(!AZ(u,v,op))continue;bool _ok=1;for(int s=0;s<2&&_ok;++s){int x=s?u:v;for(int f:vf[x])if(Ab[f]){int a=FA[f],b=FB[f],c=FC[f];if(a==b||b==c||a==c)continue;double on[3];if(!Af(X[a],Y[a],Z[a],X[b],Y[b],Z[b],X[c],Y[c],Z[c],on[0],on[1],on[2]))continue;auto _gx=[&](int w){return(w==x)?_px:X[w];};auto _gy=[&](int w){return(w==x)?_py:Y[w];};auto _gz=[&](int w){return(w==x)?_pz:Z[w];};double nn[3];if(!Af(_gx(a),_gy(a),_gz(a),_gx(b),_gy(b),_gz(b),_gx(c),_gy(c),_gz(c),nn[0],nn[1],nn[2])){_ok=0;break;}double nd=Ba(on[0],on[1],on[2],nn[0],nn[1],nn[2]);if(nd>0.00035){_ok=0;break;}}}if(!_ok)continue;X[v]=_px;Y[v]=_py;Z[v]=_pz;Ab[fs[0]]=Ab[fs[1]]=0;for(int f:vf[u])if(Ab[f]){if(FA[f]==u)FA[f]=v;if(FB[f]==u)FB[f]=v;if(FC[f]==u)FC[f]=v;vf[v].push_back(f);}Ag[u]=0;uf[u]=v;--A3;++_fe;if(vf[v].size()>64){vector<int>_cc;_cc.reserve(32);for(int f:vf[v])if(Ab[f])_cc.push_back(f);vf[v].swap(_cc);}break;}}if(_fe==0)goto _ed4;}_ed4:;}
if(NV>10000&&NV<=30000){int _dc=0;for(int _p=0;_p<5;++_p){for(int v=0;v<NV;++v)if(Ag[v]){vector<int>_nb;As(v,_nb);for(int u:_nb)if(Ag[u]&&u>v){int fs[2],op[2];if(Ar(u,v,fs,op)!=2)continue;double _n0[3],_n1[3];{int a=FA[fs[0]],b=FB[fs[0]],c=FC[fs[0]];if(!Af(X[a],Y[a],Z[a],X[b],Y[b],Z[b],X[c],Y[c],Z[c],_n0[0],_n0[1],_n0[2]))continue;}{int d=FA[fs[1]],e=FB[fs[1]],f=FC[fs[1]];if(!Af(X[d],Y[d],Z[d],X[e],Y[e],Z[e],X[f],Y[f],Z[f],_n1[0],_n1[1],_n1[2]))continue;}if(_n0[0]*_n1[0]+_n0[1]*_n1[1]+_n0[2]*_n1[2]<0.995)continue;double _px=(X[u]+X[v])*.5,_py=(Y[u]+Y[v])*.5,_pz=(Z[u]+Z[v])*.5;if(!AZ(u,v,op))continue;bool _ok=1;for(int s=0;s<2&&_ok;++s){int x=s?u:v;for(int f:vf[x])if(Ab[f]){int a=FA[f],b=FB[f],c=FC[f];if(a==b||b==c||a==c)continue;double on[3];if(!Af(X[a],Y[a],Z[a],X[b],Y[b],Z[b],X[c],Y[c],Z[c],on[0],on[1],on[2]))continue;auto _gx=[&](int w){return(w==x)?_px:X[w];};auto _gy=[&](int w){return(w==x)?_py:Y[w];};auto _gz=[&](int w){return(w==x)?_pz:Z[w];};double nn[3];if(!Af(_gx(a),_gy(a),_gz(a),_gx(b),_gy(b),_gz(b),_gx(c),_gy(c),_gz(c),nn[0],nn[1],nn[2])){_ok=0;break;}if(Ba(on[0],on[1],on[2],nn[0],nn[1],nn[2])>0.00080){_ok=0;break;}}}if(!_ok)continue;X[v]=_px;Y[v]=_py;Z[v]=_pz;Ab[fs[0]]=Ab[fs[1]]=0;for(int f:vf[u])if(Ab[f]){if(FA[f]==u)FA[f]=v;if(FB[f]==u)FB[f]=v;if(FC[f]==u)FC[f]=v;vf[v].push_back(f);}Ag[u]=0;uf[u]=v;--A3;++_dc;if(vf[v].size()>64){vector<int>_cc;_cc.reserve(32);for(int f:vf[v])if(Ab[f])_cc.push_back(f);vf[v].swap(_cc);}break;}}if(_dc==0)goto _y3;}_y3:;}
if(NV>30000&&NV<=45000){int _dc=0;for(int _p=0;_p<8;++_p){for(int v=0;v<NV;++v)if(Ag[v]){vector<int>_nb;As(v,_nb);for(int u:_nb)if(Ag[u]&&u>v){int fs[2],op[2];if(Ar(u,v,fs,op)!=2)continue;double _n0[3],_n1[3];{int a=FA[fs[0]],b=FB[fs[0]],c=FC[fs[0]];if(!Af(X[a],Y[a],Z[a],X[b],Y[b],Z[b],X[c],Y[c],Z[c],_n0[0],_n0[1],_n0[2]))continue;}{int d=FA[fs[1]],e=FB[fs[1]],f=FC[fs[1]];if(!Af(X[d],Y[d],Z[d],X[e],Y[e],Z[e],X[f],Y[f],Z[f],_n1[0],_n1[1],_n1[2]))continue;}if(_n0[0]*_n1[0]+_n0[1]*_n1[1]+_n0[2]*_n1[2]<0.999)continue;double _px=(X[u]+X[v])*.5,_py=(Y[u]+Y[v])*.5,_pz=(Z[u]+Z[v])*.5;if(!AZ(u,v,op))continue;bool _ok=1;for(int s=0;s<2&&_ok;++s){int x=s?u:v;for(int f:vf[x])if(Ab[f]){int a=FA[f],b=FB[f],c=FC[f];if(a==b||b==c||a==c)continue;double on[3];if(!Af(X[a],Y[a],Z[a],X[b],Y[b],Z[b],X[c],Y[c],Z[c],on[0],on[1],on[2]))continue;auto _gx=[&](int w){return(w==x)?_px:X[w];};auto _gy=[&](int w){return(w==x)?_py:Y[w];};auto _gz=[&](int w){return(w==x)?_pz:Z[w];};double nn[3];if(!Af(_gx(a),_gy(a),_gz(a),_gx(b),_gy(b),_gz(b),_gx(c),_gy(c),_gz(c),nn[0],nn[1],nn[2])){_ok=0;break;}if(Ba(on[0],on[1],on[2],nn[0],nn[1],nn[2])>0.00050){_ok=0;break;}}}if(!_ok)continue;X[v]=_px;Y[v]=_py;Z[v]=_pz;Ab[fs[0]]=Ab[fs[1]]=0;for(int f:vf[u])if(Ab[f]){if(FA[f]==u)FA[f]=v;if(FB[f]==u)FB[f]=v;if(FC[f]==u)FC[f]=v;vf[v].push_back(f);}Ag[u]=0;uf[u]=v;--A3;++_dc;if(vf[v].size()>64){vector<int>_cc;_cc.reserve(32);for(int f:vf[v])if(Ab[f])_cc.push_back(f);vf[v].swap(_cc);}break;}}if(_dc==0)goto _y4;}_y4:;}
vector<int> H5(NV,-1);
int Ke=0;
for(int i=0;i <NV;++i)if(Ag[i])H5[i]=Ke++;
int Kf=0;
for(int f=0;f <NF;++f){if(!Ab[f]&&!Dn(f))continue;
int a=FA[f],b=FB[f],c=FC[f];
if(a==b||b==c||a==c)continue;
++Kf;}
string Ip;
Ip.reserve((size_t)Ke*40+(size_t)Kf*24+32);
char Gx[96];
Ip.append(Gx,snprintf(Gx,sizeof Gx,"%d %d\n",Ke,Kf));
for(int i=0;i <NV;++i){if(!Ag[i])continue;
Ip.append(Gx,snprintf(Gx,sizeof Gx,"v %.10g %.10g %.10g\n",X[i],Y[i],Z[i]));}
for(int f=0;f <NF;++f){if(!Ab[f]&&!Dn(f))continue;
int a=FA[f],b=FB[f],c=FC[f];
if(a==b||b==c||a==c)continue;
Ip.append(Gx,snprintf(Gx,sizeof Gx,"f %d %d %d\n",H5[a]+1,H5[b]+1,H5[c]+1));}
fwrite(Ip.data(),1,Ip.size(),stdout);
fflush(stdout);
_Exit(0);}