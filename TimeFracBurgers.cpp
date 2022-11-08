#include "StochasticENKF.hpp"
#include <stdio.h>
#include <string>
#include <iostream>

using namespace arma;

class Mesh{
    vec mMeshPoints;
    sp_mat mIntegralMat;
    sp_mat mDerivativeMat;

public:
    Mesh(vec meshPoints): mMeshPoints(meshPoints){
        int n = meshPoints.size()-2;
        mIntegralMat = sp_mat(n, n);
        mDerivativeMat = sp_mat(n,n);
        // 下面是初始化
        mIntegralMat(0,0) = (mMeshPoints[2] - mMeshPoints[0]) / 2.0;
        mIntegralMat(0,1) = (mMeshPoints[2] - mMeshPoints[1]) / 6.0;
        for(int i=1; i<n-1; i++){
            mIntegralMat(i,i-1) = (mMeshPoints[i+1] - mMeshPoints[i]) / 6.0;
            mIntegralMat(i,i) = (mMeshPoints[i+2] - mMeshPoints[i]) / 2.0;
            mIntegralMat(i,i+1) = (mMeshPoints[i+2] - mMeshPoints[i+1]) / 6.0;
        }
        mIntegralMat(n-1,n-2) = (mMeshPoints[n] - mMeshPoints[n-1]) / 6.0;
        mIntegralMat(n-1,n-1) = (mMeshPoints[n+1] - mMeshPoints[n-1]) / 2.0;
        // 下面是导数积分矩阵初始化
        mDerivativeMat(0,0) = 1.0 / (mMeshPoints[1] - mMeshPoints[0]) +
                            1.0 / (mMeshPoints[2] - mMeshPoints[1]);
        mDerivativeMat(0,1) = -1.0 / (mMeshPoints[2] - mMeshPoints[1]);
        for(int i=1; i<n-1; i++){
            mDerivativeMat(i,i-1) = -1.0/(mMeshPoints[i+1] - mMeshPoints[i]);
            mDerivativeMat(i,i) = 1.0 / (mMeshPoints[i+1] - mMeshPoints[i]) +
                                1.0/(mMeshPoints[i+2] - mMeshPoints[i+1]);
            mDerivativeMat(i,i+1) = -1.0/(mMeshPoints[i+2] - mMeshPoints[i+1]);
        }
        mDerivativeMat(n-1, n-2) = -1.0/(mMeshPoints[n] - mMeshPoints[n-1]);
        mDerivativeMat(n-1,n-1) = 1.0/(mMeshPoints[n+1] - mMeshPoints[n]) +
                                1.0 / (mMeshPoints[n] - mMeshPoints[n-1]);
    }

    const vec& getMeshPoints() {return mMeshPoints;}
    const sp_mat& getIntegralMat() {return mIntegralMat;}
    const sp_mat& getDerivativeMat() {return mDerivativeMat;}
};


class Burgers{
    double alpha;
    Mesh& mMesh;
    double mCAlphaDeltaT;
    int N;
    double v;
    double mDeltaT;

public:
    Burgers(double _alpha, double _v, Mesh& mesh, double deltaT)
    :alpha(_alpha), mMesh(mesh), N(mesh.getIntegralMat().n_cols), v(_v), mDeltaT(deltaT){
        mCAlphaDeltaT = pow(deltaT, -_alpha) / tgamma(2 - _alpha); 
    }

    static vec 
    nonlinearSolver
    (double cAlphaDeltaT, vec f, double v, Mesh& mesh, double epsilon=1e-15, int maxTimes=100){
        int count=0;

        vec b(f.size(), arma::fill::zeros);
        vec newSol(f.size(), arma::fill::zeros);
        vec oldSol(f.size(), arma::fill::ones);

        // 初始化b
        for(int j=0; j<b.size(); j++)
            for(int i= (j-1>=0?j-1:0); i<=j+1 && i<b.size(); i++)
                b[j] -= f[i]*mesh.getIntegralMat()(i,j);
        // 循环求解，直到改进很小
        while(max(abs(newSol - oldSol)) > epsilon && count < maxTimes){
            count++;
            oldSol = newSol;
            sp_mat a(f.size(), f.size());
            // 初始化a,这里j是行的下标
            for(int j=0; j<a.n_rows; j++){
                if(j-1 >= 0)
                    a(j,j-1) = cAlphaDeltaT*mesh.getIntegralMat()(j,j-1) + v*mesh.getDerivativeMat()(j,j-1) \
                                - 0.5*(oldSol[j-1]/3.0 + oldSol[j]/6.0);
                a(j,j) = cAlphaDeltaT*mesh.getIntegralMat()(j,j) + v*mesh.getDerivativeMat()(j,j) -
                            0.5*(oldSol[j-1]/6.0 - oldSol[j+1]/6.0);
                if(j+1 < a.n_cols)
                    a(j,j+1) = cAlphaDeltaT*mesh.getIntegralMat()(j,j+1) + v*mesh.getDerivativeMat()(j,j+1) -
                            0.5*(-oldSol[j]/6.0 - oldSol[j+1]/3.0);
            }
            newSol = spsolve(a, b);
        }
        // printf("计算%d次\n",count);
        // 求解完毕
        return newSol;
    }

    mat forward(mat& ensemble, int idx, mat& sysVar){
        // printf("%d\n", idx);
        int n = ensemble.n_rows / this->N;
        vec bAlpha = BAlpha(this->alpha, n+2);
        mat sols(N, ensemble.n_cols);
        
        for(int j=0; j<ensemble.n_cols; j++){
            // 约定时间近的在下面
            vec formerInfo(ensemble.submat(0,j,N-1,j));
            formerInfo *= - bAlpha[n-1];
            for(int i=1; i<n; i++){
                vec temp(ensemble.submat((n-i)*N,j,(n-i+1)*N-1,j));
                formerInfo += (bAlpha[i] - bAlpha[i-1]) * temp;
            }
            formerInfo *= this->mCAlphaDeltaT;
            vec sol = nonlinearSolver(this->mCAlphaDeltaT, formerInfo, v, mMesh);
            sols.col(j) = sol;
        }
        
        sols += mvnrnd(zeros(N), sysVar, ensemble.n_cols);
        return join_cols(ensemble, sols);
    }

    mat inverse(mat& ensemble, int idx, mat& sysVar){
        // printf("inverse: %d\n", idx);

        int n = (ensemble.n_rows - 1) / this->N;

        mat sols(this->N + 1, ensemble.n_cols);

        for(int j=0; j<ensemble.n_cols; j++){
            // printf("inverse column %d\n", j);
            // 下面是每一个成员特有的需要计算的数据
            double ensembleAlpha = ensemble(ensemble.n_rows-1,j);
            vec bAlpha = BAlpha(ensembleAlpha, n+2);
            double ensembleCAlphaDelta = pow(mDeltaT, -ensembleAlpha) / tgamma(2-ensembleAlpha);

            vec formerInfo(ensemble.submat(0,j,N-1,j));
            formerInfo *= - bAlpha[n-1];
            for(int i=1; i<n; i++){
                vec temp(ensemble.submat((n-i)*N,j,(n-i+1)*N-1,j));
                formerInfo += (bAlpha[i] - bAlpha[i-1]) * temp;
            }
            formerInfo *= ensembleCAlphaDelta;
            vec sol = nonlinearSolver(ensembleCAlphaDelta, formerInfo, v, mMesh);
            sols.submat(0,j,sols.n_rows-2,j) = sol;
            sols(sols.n_rows-1,j) = ensembleAlpha;
        } 

        sols += mvnrnd(zeros(N+1), sysVar, ensemble.n_cols);

        return join_cols(ensemble.submat(0,0,ensemble.n_rows-2,ensemble.n_cols-1), sols);
    }
};


vec init(const vec& x){
    return arma::sin(datum::pi * x);
}


// 该函数接受alpha，返回计算结果并保存在指定的文件路径中
mat burgersTest(double alpha, std::string filename){
    vec xMesh = arma::linspace(0,2,401);
    vec u0 = init(xMesh).subvec(1,399);
    mat sol(u0);
    mat sysVar(u0.size(), u0.size(), arma::fill::zeros);
    double deltaT = 0.001;
    Mesh mesh(xMesh);

    Burgers burgers(alpha, 0.01/datum::pi, mesh, deltaT);
    for(int i=0; i<(int)(1.0/deltaT); i++){
        // printf("第%d次迭代\n", i);
        sol = burgers.forward(sol,i,sysVar);
    }
    sol.reshape(xMesh.size()-2, sol.n_rows/(xMesh.size()-2));
    mat t=sol.t();
    t.save(filename, arma::raw_ascii);
    return t;
}


mat fracObOp(const mat& input){
    /*
    mat obResult(4, input.n_cols);
    for(int i=0; i<4; i++)
        obResult.row(4-1-i) = input.row(input.n_rows - 80 - i*80);
    */
    mat obResult(9, input.n_cols);
    for(int i=0; i<9; i++)
        obResult.row(9-1-i) = input.row(input.n_rows - 40 - i*40);
    return obResult;
}


mat inverseFracObOp(const mat& input){
    mat noAlpha = input.submat(0, 0, input.n_rows-2, input.n_cols-1);
    return fracObOp(noAlpha);
}

void fracBurgersENKFTest(double alpha){
    // 生成参考解
    std::string filename = std::to_string(alpha);
    filename += "_burgers.csv";
    mat sol = burgersTest(alpha, filename);
    printf("计算完参考解\n");

    // 基本参数
    double v = 0.01 / arma::datum::pi;
    double deltaT = 0.001;
    double obVar = 0.01;
    double initVar = 0.01;
    int iters = 1/deltaT+1;
    vec xMesh = arma::linspace(0,2,401);
    Mesh mesh(xMesh);
    Burgers burgers(alpha, v, mesh, deltaT);

    auto wrapper = [&burgers](mat& ensembleAnalysis, int idx, mat& sysVar)
                            -> mat{
                                //printf("第%d次\n", idx);
                                return burgers.forward(ensembleAnalysis, idx, sysVar);
                            };
    
    // 填充观测值
    std::vector<vec> obLists;
    for(int i=0; i<iters; i++)
        obLists.push_back(vec());
    
    std::vector<int> obPlaces{0,100,200,300,400,500,600,700,800,900,1000};
    for(int& place: obPlaces){
        obLists[place] = fracObOp(
            sol.submat(place, 0, place, sol.n_cols-1).t()
            );
        obLists[place] += sqrt(obVar) * arma::randn(obLists[place].n_elem);
    }

    int obSize = obLists[obPlaces[0]].n_rows;
    auto obErrorPtr = std::make_shared<mat>(obSize, obSize, arma::fill::eye);
    *obErrorPtr *= obVar; 

    printf("ob ready\n");

    /*
    // 获得ENKF需要的参数
    // 观测列表
    std::vector<vec> obLists;
    for(int i=0; i<4; i++){
        obLists.push_back(ob.col(i));
        for(int j=0; j<24; j++)
            obLists.push_back(vec());
    }
    obLists.push_back(ob.col(4));
    printf("obList ready\n");
    */
    // 错误矩阵
    Errors obErrors;
    for(int i=0; i<obLists.size(); i++)
        obErrors.add(obErrorPtr);
    // 观测算子
    ObserveOperator obOp = fracObOp;
    printf("obOp ready\n");
    // 系统误差
    auto sysVarPtr = std::make_shared<mat>(xMesh.size()-2, xMesh.size()-2, arma::fill::eye);
    *sysVarPtr *= 0.0001;
    Errors sysErrors;
    for(int i=0; i<obLists.size(); i++)
        sysErrors.add(sysVarPtr);
    printf("sysError ready\n");
    // 初始值
    vec initAve = init(xMesh.subvec(1,399));
    initAve += sqrt(initVar)*arma::randn(initAve.size());
    mat initVarMat = initVar*arma::eye(initAve.size(), initAve.size());
    // ENKF
    int ensembleSize = 20;
    printf("ready\n");

    mat forAdd(iters, initAve.size(), arma::fill::zeros);
    int numENKF=1;
    for(int i=0; i<numENKF; i++){
        printf("第%d次ENKF\n", i);
        try{
        auto result = StochasticENKF(ensembleSize, initAve, initVarMat, obLists, iters, obErrors, obOp, wrapper, sysErrors);
        auto last = result.back(); 
        mat lastMat = reshape(last, initAve.size(), last.size()/initAve.size()).t();
        forAdd += lastMat;
        }catch(std::runtime_error e){
            numENKF--;
        }
    }
    forAdd /= numENKF;
    forAdd.save("analysis"+filename, arma::raw_ascii);
}

// 反问题
void inverseFracBurgersENKFTest(double alpha){
    // 生成参考解
    std::string filename = std::to_string(alpha);
    filename += "_burgers.csv";
    mat sol = burgersTest(alpha, filename);
    printf("计算完参考解\n");

    // 基本参数
    double v = 0.01 / arma::datum::pi;
    double deltaT = 0.001;
    double obVar = 0.01;
    double initVar = 0.01;
    int iters = 1/deltaT+1;
    vec xMesh = arma::linspace(0,2,401);
    Mesh mesh(xMesh);
    Burgers burgers(alpha, v, mesh, deltaT);

    auto wrapper = [&burgers](mat& ensembleAnalysis, int idx, mat& sysVar)
                            -> mat{
                                //printf("第%d次\n", idx);
                                return burgers.inverse(ensembleAnalysis, idx, sysVar);
                            };
    
    // 填充观测值
    std::vector<vec> obLists;
    for(int i=0; i<iters; i++)
        obLists.push_back(vec());
    
    std::vector<int> obPlaces{0,100,200,300,400,500,600,700,800,900,1000};
    for(int& place: obPlaces){
        obLists[place] = fracObOp(
            sol.submat(place, 0, place, sol.n_cols-1).t()
            );
        obLists[place] += sqrt(obVar) * arma::randn(obLists[place].n_elem);
    }

    int obSize = obLists[obPlaces[0]].n_rows;
    auto obErrorPtr = std::make_shared<mat>(obSize, obSize, arma::fill::eye);
    *obErrorPtr *= obVar; 

    printf("ob ready\n");

    // 错误矩阵
    Errors obErrors;
    for(int i=0; i<obLists.size(); i++)
        obErrors.add(obErrorPtr);
    // 观测算子
    ObserveOperator obOp = inverseFracObOp;
    printf("obOp ready\n");
    // 系统误差
    auto sysVarPtr = std::make_shared<mat>(xMesh.size()-2+1, xMesh.size()-2+1, arma::fill::eye);
    *sysVarPtr *= 0.0001;
    Errors sysErrors;
    for(int i=0; i<obLists.size(); i++)
        sysErrors.add(sysVarPtr);
    printf("sysError ready\n");
    // 初始值
    vec initAve = init(xMesh.subvec(1,399+1));
    //initAve += sqrt(initVar)*arma::randn(initAve.size());
    initAve(399) = alpha;// + sqrt(initVar)*arma::randn();
    mat initVarMat = initVar*arma::eye(initAve.size(), initAve.size());
    printf("init alpha: %f\n", initAve(399));
    // ENKF
    int ensembleSize = 20;
    printf("ready\n");

    mat forAdd(iters, initAve.size()-1, arma::fill::zeros);
    vec forAllAlpha(iters, arma::fill::zeros);
    vec forAlpha(1, arma::fill::zeros);
    // 多次ENKF
    int numENKF=1;
    int realTimes=numENKF;
    for(int i=0; i<numENKF; i++){
        printf("第%d次ENKF\n", i);
        try{
        auto result = StochasticENKF(ensembleSize, initAve, initVarMat, obLists, iters, obErrors, obOp, wrapper, sysErrors);
        
        // 想要看到alpha的计算过程
        for(int i=0; i<result.size(); i++){
            forAllAlpha(i) += result[i](result[i].size()-1);
        }


        // 处理结果
        auto last = result.back(); 
        mat lastMat = arma::reshape(last.subvec(0,last.size()-2), 
                            initAve.size()-1, 
                            (last.size()-2)/(initAve.size()-1)
                            ).t();
        forAdd += lastMat;

        forAlpha += last(last.size()-1);

        }catch(std::exception& e){
            std::cout<<"something wrong\n";
            realTimes--;
        }
    }

    // 保存
    forAdd /= realTimes;
    forAlpha /= realTimes;
    forAllAlpha /= realTimes;
    forAdd.save("analysis"+filename, arma::raw_ascii);
    forAlpha.save("alpha"+filename, arma::raw_ascii);
    forAllAlpha.save("all_alpha"+filename, arma::raw_ascii);
    std::cout<<forAllAlpha<<'\n';
    std::cout<<forAlpha;
}


int main(int argc, char** argv){
    if(argc <= 1){
        printf("usage: %s <alpha>\n", argv[0]);
        exit(1);
    }
    double alpha = std::stod(argv[1]);
    fracBurgersENKFTest(alpha);
}