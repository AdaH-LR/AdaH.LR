clear all
clc
addpath('./Evaluation function')
addpath('./glmnet')

load('D:\ziliao\daima\shiyandaima\data\GSE10072\GSE10072_Path_GO');
Data=Data';
Lable=Label';
% Data=log2(Data+1);

[samples,features]=size(Data);
[P_num,~]=size(GSE10072_Path);
Data=standardizeCols(Data);

fold=5;
L_beta=[];
AL_beta=[];
M_beta=[];
PS_beta=[];
WEN_beta=[];
EN_beta=[];
Lasso_beta=[];
AH_beta=[];
AWH_beta=[];
AWHRM_beta=[];
MNet1_beta=[]; 
MNet2_beta=[];  
MNet3_beta=[]; 
%%
n=3;
for i=1:n
    tic
    fprintf('cycle:%d \n',i);
    nselect=round(samples/2);
    select=round(randperm(samples,nselect));
    X_train=Data(select,:);
    Y_train=Lable(select);
    X_test=[];
    Y_test=[];
    for p=1:samples
        in=0;
        for q=1:nselect
            k=select(q);
            if(p==k)
                in=1;
                break;
            end
        end
        if(in==0)
            X_test=[X_test;Data(p,:)];
            Y_test=[Y_test;Lable(p)];
        end
    end
    %%
    
    tic
    [H,~,A,B,L,Lh,L1,L2,L3]=get_laplacin(X_train,GSE10072_Path,SIM,Union_SIM);
    toc
    sigma=0.5
    fprintf('\n\n');
    %%
    %Net
%     lambda_l=[0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001];
%     alpha_l=[0.1,0.05,0.02,0.01,0.005,0.001,0.0001,0.00001];
% %     lambda_l=[0.1,0.05];
% %     alpha_l=[0.001,0.0001,0.00001];
%     for j=1:length(lambda_l)
%         for k=1:length(alpha_l)
%             index=(j-1)*length(alpha_l)+k;
%             alpha=alpha_l(k);
%             lambda=lambda_l(j); 
%             [beta,intercept]=Net_Logistics(X_train,Y_train,L,alpha,lambda);
%             [YI,F1,AUC,predict_y]=Evalution_function(X_test,Y_test,beta,intercept);
%             [ YI ]=Youden_index( Y_test,predict_y )
%             L_beta=[L_beta,beta];
%             Log_L(index,i)=YI;
%             Log_l(index,i)=AUC;
%             Log_l=F1
%             L_num(index,i)=sum(beta~=0);
%              
%             fprintf('%dL_logisitc regression: peformance(Youden Index)=%f (AUC)=%f (num)=%d\n\n',index,YI,AUC,L_num(index,i));
%         end
%     end

    %%
    %AbsNet
%     lambda_al=[0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001];                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
%     alpha_al=[0.1,0.05,0.02,0.01,0.005,0.001,0.0001,0.00001];
% %     lambda_al=[0.1,0.05];
% %     alpha_al=[0.001,0.0001,0.00001];
%     for j=1:length(lambda_al)
%         for k=1:length(alpha_al)
%             index=(j-1)*length(alpha_al)+k;
%             alpha=alpha_al(k);
%             lambda=lambda_al(j); 
%             [beta,intercept]=Abs_Net_Logistics(X_train,Y_train,L,alpha,lambda);
%             [YI,F1,AUC,predict_y]=Evalution_function(X_test,Y_test,beta,intercept);
%             [ YI ]=Youden_index( Y_test,predict_y )
%             AL_beta=[AL_beta,beta];
%             Log_AL(index,i)=YI;
%             Log_al(index,i)=AUC;
%             Log_al=F1
%             AL_num(index,i)=sum(beta~=0);
%              
%             fprintf('%dAL_logisitc regression: peformance(Youden Index)=%f (AUC)=%f  (num)=%d\n\n',index,YI,AUC,AL_num(index,i));
%         end
%     end
    
    %%
    %Modularity
    lambda_m=[0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001];
    alpha_m=[0.01,0.005,0.001,0.0001,0.00001];
%     lambda_m=[0.1,0.05];
%     alpha_m=[0.001,0.0001,0.00001];
    for j=1:length(lambda_m)
        for k=1:length(alpha_m)
            index=(j-1)*length(alpha_m)+k;
            alpha=alpha_m(k);
            lambda=lambda_m(j); 
            [beta,intercept]=M_Logistics(X_train,Y_train,B,alpha,lambda);
            [YI,F1,AUC,predict_y]=Evalution_function(X_test,Y_test,beta,intercept);
            [ YI ]=Youden_index( Y_test,predict_y )
            M_beta=[M_beta,beta];
            Log_M(index,i)=YI;
            Log_m(index,i)=AUC;
            Log_m=F1
            M_num(index,i)=sum(beta~=0);
             
            fprintf('%dM_logisitc regression: peformance(Youden Index)=%f (AUC)=%f  (num)=%d\n\n',index,YI,AUC,M_num(index,i));
        end
    end
    
    %%
    %Structured Penalized
    lambda_ps=[0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001];
    alpha_ps=[0.1,0.05,0.02,0.01,0.005,0.001,0.0001,0.00001];
%     lambda_ps=[0.1,0.05];
%     alpha_ps=[0.02,0.01,0.005];
    for j=1:length(lambda_ps)
        for k=1:length(alpha_ps)
            index=(j-1)*length(alpha_ps)+k;
            alpha=alpha_ps(k);
            lambda=lambda_ps(j); 
            [beta,intercept]=PS_Logistics(X_train,Y_train,alpha,lambda);
            [YI,F1,AUC,predict_y]=Evalution_function(X_test,Y_test,beta,intercept);
            [ YI ]=Youden_index( Y_test,predict_y )
            PS_beta=[PS_beta,beta];
            Log_PS(index,i)=YI;
            Log_ps(index,i)=AUC;
            Log_ps=F1
            PS_num(index,i)=sum(beta~=0);
             
            fprintf('%dPS_logisitc regression: peformance(Youden Index)=%f (AUC)=%f  (num)=%d\n\n',index,YI,AUC,PS_num(index,i));
        end
    end
    
    %%
    %Weight_EN
    lambda_wen=[0.01,0.005,0.002,0.001,0.0001];
    alpha_wen=[0.1,0.05,0.02,0.01,0.005,0.001,0.0001,0.00001];
%     lambda_wen=[0.0001,0.00001];
%     alpha_wen=[0.1,0.05,0.02,0.01,0.005,0.001,0.0001,0.00001];
    for j=1:length(lambda_wen)
        for k=1:length(alpha_wen)
            index=(j-1)*length(alpha_wen)+k;
            alpha=alpha_wen(k);
            lambda=lambda_wen(j); 
            [beta,intercept]=WEN_Logistics(X_train,Y_train,alpha,lambda);
            [YI,F1,AUC,predict_y]=Evalution_function(X_test,Y_test,beta,intercept);
            [ YI ]=Youden_index( Y_test,predict_y )
            WEN_beta=[WEN_beta,beta];
            Log_WEN(index,i)=YI;
            Log_wen(index,i)=AUC;
            Log_wen=F1
            WEN_num(index,i)=sum(beta~=0);
             
            fprintf('%dWEN_logisitc regression: peformance(Youden Index)=%f (AUC)=%f  (num)=%d\n\n',index,YI,AUC,WEN_num(index,i));
        end
    end
    
    %%
    %Abs_Hypergraph
%     lambda_ah=[0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001];
%     alpha_ah=[0.1,0.05,0.02,0.01,0.005,0.001,0.0001,0.00001];
% %     lambda_ah=0.1;
% %     alpha_ah=0.001;
%     for j=1:length(lambda_ah)
%         for k=1:length(alpha_ah)
%             index=(j-1)*length(alpha_ah)+k;
%             alpha=alpha_ah(k);
%             lambda=lambda_ah(j); 
%             [beta,intercept]=Abs_H_Logistics(X_train,Y_train,Lh,alpha,lambda);
%             [YI,~,AUC]=Evalution_function(X_test,Y_test,beta,intercept);
%     
%             AH_beta=[AH_beta,beta];
%             Log_AH(index,i)=YI;
%             Log_ah(index,i)=AUC;
%             AH_num(index,i)=sum(beta~=0);
%              
%             fprintf('%dAH_logisitc regression: peformance(Youden Index)=%f (AUC)=%f  (num)=%d\n\n',index,YI,AUC,AH_num(index,i));
%         end
%     end
    
    %%
%     %Abs_Weight_Hypergraph
%     lambda_awh=[0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001];
%     alpha_awh=[0.1,0.05,0.02,0.01,0.005,0.001,0.0001,0.00001];
%     lambda_awh=[0.05,0.02,0.01,0.005];
%     alpha_awh=[0.005,0.001,0.0001,0.00001];
%     for j=1:length(lambda_awh)
%         for k=1:length(alpha_awh)
%             index=(j-1)*length(alpha_awh)+k;
%             alpha=alpha_awh(k);
%             lambda=lambda_awh(j); 
%             [beta,intercept]=Ada_H_Logistics(X_train,Y_train,Lh,alpha,lambda);
%             [YI,~,AUC]=Evalution_function(X_test,Y_test,beta,intercept);
%     
%             AWH_beta=[AWH_beta,beta];
%             Log_AWH(index,i)=YI;
%             Log_awh(index,i)=AUC;
%             AWH_num(index,i)=sum(beta~=0);
%              
%             fprintf('%dAWH_logisitc regression: peformance(Youden Index)=%f (AUC)=%f  (num)=%d\n\n',index,YI,AUC,AWH_num(index,i));
%         end
%     end
    
    %%
%     Abs_Weight_Hypergraph_Redundancy
    lambda_awhrm=[0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001];
    alpha_awhrm=[0.1,0.05,0.02,0.01,0.005,0.001,0.0001,0.00001];
    sigma_awhrm=[0.1,0.05,0.02,0.01,0.005,0.001];
%     lambda_awhrm=[0.05,0.02];
%     alpha_awhrm=[0.001,0.0001,0.00001];
%     sigma_awhrm=[0.1,0.05,0.02,0.01,0.005,0.001];
%     lambda_awhrm=0.001;
%     alpha_awhrm=0.00001;
%     sigma_awhrm=0.02;
    for j=1:length(lambda_awhrm)
        for k=1:length(alpha_awhrm)
            for l=1:length(sigma_awhrm)
                index=(j-1)*length(alpha_awhrm)*length(sigma_awhrm)+(k-1)*length(sigma_awhrm)+l;
                alpha=alpha_awhrm(k);
                lambda=lambda_awhrm(j); 
                sigma=sigma_awhrm(l);
                [beta,intercept]=Ada_HRM_Logistics(X_train,Y_train,Lh,A,alpha,lambda,sigma);
                [YI,F1,AUC,predict_y]=Evalution_function(X_test,Y_test,beta,intercept);
                [ YI ]=Youden_index( Y_test,predict_y )
                AWHRM_beta=[AWHRM_beta,beta];
                Log_AWHRM(index,i)=YI;
                Log_awhrm(index,i)=AUC;
                Log_awhrm=F1
                AWHRM_num(index,i)=sum(beta~=0);
             
                fprintf('%dAWHRM_logisitc regression: peformance(Youden Index)=%f (AUC)=%f  (num)=%d\n\n',index,YI,AUC,AWHRM_num(index,i));
            end
        end
    end
    %%
   
%     lambda_mnet1=[0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001];
%     alpha_mnet1=[0.1,0.05,0.02,0.01,0.005,0.001,0.0001,0.00001];
% %     lambda_mnet1=[0.02,0.01,0.005];
% %     alpha_mnet1=[0.0001,0.00001];
%     for j=1:length(lambda_mnet1)
%         for k=1:length(alpha_mnet1)
%             index=(j-1)*length(alpha_mnet1)+k;
%             alpha=alpha_mnet1(k);
%             lambda=lambda_mnet1(j); 
%             [beta,intercept]=WNet_Logistics(X_train,Y_train,L1,alpha,lambda);
%             [YI,~,AUC]=Evalution_function(X_test,Y_test,beta,intercept);
%     
%             MNet1_beta=[MNet1_beta,beta];
%             Log_MNet1(index,i)=YI;
%             Log_mnet1(index,i)=AUC;
%             MNet1_num(index,i)=sum(beta~=0);
%              
%             fprintf('%dMNet1_logisitc regression: peformance(Youden Index)=%f (AUC)=%f  (num)=%d\n\n',index,YI,AUC,MNet1_num(index,i));
%         end
%     end
    
    %%
    
%     lambda_mnet2=[0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001];
%     alpha_mnet2=[0.1,0.05,0.02,0.01,0.005,0.001,0.0001,0.00001];
%     lambda_mnet2=[0.1,0.05];
%     alpha_mnet2=[0.1,0.05,0.02,0.01,0.005,0.001,0.0001,0.00001];
%     for j=1:length(lambda_mnet2)
%         for k=1:length(alpha_mnet2)
%             index=(j-1)*length(alpha_mnet2)+k;
%             alpha=alpha_mnet2(k);
%             lambda=lambda_mnet2(j);
%             [beta,intercept]=WNet_Logistics(X_train,Y_train,L2,alpha,lambda);
%             [YI,~,AUC]=Evalution_function(X_test,Y_test,beta,intercept);
%     
%             MNet2_beta=[MNet2_beta,beta];
%             Log_MNet2(index,i)=YI;
%             Log_mnet2(index,i)=AUC;
%             MNet2_num(index,i)=sum(beta~=0);
%              
%             fprintf('%dMNet2_logisitc regression: peformance(Youden Index)=%f (AUC)=%f  (num)=%d\n\n',index,YI,AUC,MNet2_num(index,i));
%         end
%     end
    
    %%
  
%     lambda_mnet3=[0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001];
%     alpha_mnet3=[0.1,0.05,0.02,0.01,0.005,0.001,0.0001,0.00001];
% %     lambda_mnet3=[0.1,0.05];
% %     alpha_mnet3=[0.0001,0.00001];
%     for j=1:length(lambda_mnet3)
%         for k=1:length(alpha_mnet3)
%             index=(j-1)*length(alpha_mnet3)+k;
%             alpha=alpha_mnet3(k);
%             lambda=lambda_mnet3(j); 
%             [beta,intercept]=WNet_Logistics(X_train,Y_train,L3,alpha,lambda);
%             [YI,~,AUC]=Evalution_function(X_test,Y_test,beta,intercept);
%     
%             MNet3_beta=[MNet3_beta,beta];
%             Log_MNet3(index,i)=YI;
%             Log_mnet3(index,i)=AUC;
%             MNet3_num(index,i)=sum(beta~=0);
%              
%             fprintf('%dMNet3_logisitc regression: peformance(Youden Index)=%f (AUC)=%f  (num)=%d\n\n',index,YI,AUC,MNet3_num(index,i));
%         end
%     end
    
    %%
    %%%%%%%%%%%%%%%%%%%%%%%%% EN_Logisitc %%%%%%%%%%%%%%%%%%%%%%%%%%
    
    options = glmnetSet;
    options.alpha = 0.5;
    %options.nlambda = 10;
    
    fit=glmnet(X_train,Y_train(:,1) + 1,'binomial',options);
    CVerr=cvglmnet(X_train,Y_train(:,1)+1,fold,[],'class','binomial',glmnetSet,0);
    beta_path_en = fit.beta;
    intercept_path = fit.a0;
    lambda_path = fit.lambda;
    opt_index = find(CVerr.glmnetOptions.lambda == CVerr.lambda_min);
    [ YI,~,AUC ] = Evalution_function( X_test,Y_test(:,1),beta_path_en(:,opt_index),intercept_path(opt_index) );
    EN_beta=beta_path_en(:,opt_index);
    Log_EN(1,i)=YI;
    Log_en(1,i)=AUC;
    EN_num(i)=sum(beta_path_en(:,opt_index)~=0);
    
    fprintf('EN_logisitc regression: peformance(Youden Index)=%f (AUC)=%f (selet_num)=%f\n\n',YI,AUC,EN_num(i));
    
    %%%%%%%%%%%%%%%%%%%%%%%%% Lasso_Logisitc %%%%%%%%%%%%%%%%%%%%%%%%%%
    
    options = glmnetSet;
    options.alpha = 1;
    %options.nlambda = 10;
    
    fit=glmnet(X_train,Y_train(:,1) + 1,'binomial',options);
    CVerr=cvglmnet(X_train,Y_train(:,1)+1,fold,[],'class','binomial',glmnetSet,0);
    beta_path_lasso = fit.beta;
    intercept_path = fit.a0;
    lambda_path = fit.lambda;
    opt_index = find(CVerr.glmnetOptions.lambda == CVerr.lambda_min);
    [ YI,~,AUC ] = Evalution_function( X_test,Y_test(:,1),beta_path_lasso(:,opt_index),intercept_path(opt_index) );
    Lasso_beta=beta_path_lasso(:,opt_index);
    Log_Lasso(1,i)=YI;
    Log_lasso(1,i)=AUC;
    Lasso_num(i)=sum(beta_path_lasso(:,opt_index)~=0);
    
    fprintf('Lasso_logisitc regression: peformance(Youden Index)=%f (AUC)=%f (selet_num)=%f\n\n',YI,AUC,Lasso_num(i));
    temp=0;
    toc
    save('beta','EN_beta','Lasso_beta');
    a=sum(Lasso_beta~=0);
    A=sum(EN_beta~=0);
end
