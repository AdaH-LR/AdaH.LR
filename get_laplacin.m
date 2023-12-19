function [H,F_Weight,A,B,L,Lh,L_SIM_temp,L_H_SIM,L_Con_SIM]=get_laplacin(X,Path,SIM,Con_SIM)
[row,col]=size(X);
oe=ones(col,1);

Path(sum(Path,2)==0,:)=[];
% Empty_Index=find(sum(Path,2)==0);

[paths,~]=size(Path);
%%
%基因通路邻接矩阵
H=zeros(col);
for i=1:col
    for j=1:paths
        if(Path(j,i)==0)
            continue;
        end
        for k=1:col
            if(Path(j,k)~=0&&i~=k)
                H(i,k)=1;
                H(k,i)=1;
            end
        end
    end
end
H=H-diag(diag(H));
%%
%统计学邻接矩阵(欧式距离）
% F_Weight=zeros(col);
% F_Weight_temp=zeros(col);
% sum_w=0;
% for i=1:col
%     for j=1:col
%         F_Weight_temp(i,j)=norm(X(:,i)-X(:,j));
%         sum_w=sum_w+F_Weight_temp(i,j);
%     end
% end
% delta=sum_w/(col^2);
% for i=1:col
%     for j=1:col
%         if i==j
%             continue;
%         end
%         F_Weight(i,j)=exp(-(F_Weight_temp(i,j))^2/delta^2);
%         if F_Weight(i,j) < 0.0 || isnan(F_Weight(i,j))==1
%             F_Weight(i,j) = 0 ;
%         end
%     end
% end
% F_Weight = (F_Weight'+F_Weight)./2;
F_Weight=0;
%%
%统计学邻接矩阵
A=abs(corr(X)).^2;
A=A-diag(diag(A));
for i = 1: col
    for j = 1: col
        if A(i,j) < 0.0 || isnan(A(i,j))==1
            A(i,j) = 0 ;
        end
    end
end
%%
%基因通路拉普拉斯矩阵
DL=diag(H*oe);
L=DL-H;
%%
%超图拉普拉斯矩阵
Weight=Path*oe;
LW=diag(Weight);
De=diag(Weight);
Dv=diag(Weight'*Path);
Lh=Dv-(Path'*LW/De*Path);
%%
%modularity矩阵
WL=sum(H);  %得到邻接矩阵以及矩阵中每个节点的度
L_num=sum(sum(H));
B=zeros(col);
for i=1:col
    for j=1:col
        if i~=j
            B(i,j)=H(i,j)-WL(i)*WL(j)/(L_num);
        end
    end
end
%%
%基因本体相似性拉普拉斯矩阵
D_SIM = diag(sum(SIM));
L_SIM = D_SIM-SIM;
%%
%基因通路+基因本体+统计学信息
% A = (1-sigma)*SIM+sigma*F_Weight;
% A(find(H==0)) = 0;
% D2 = diag(sum(A));
% L2 = D2-A;
%结合基因本体
%%
%基因通路+基因本体
SIM_temp = SIM;
SIM_temp(find(H==0))=0;
D_SIM_temp = diag(sum(SIM_temp));
L_SIM_temp = D_SIM_temp-SIM_temp;
%%
%超图拉普拉斯矩阵+基因本体拉普拉斯矩阵
sigma=0.5;
I=eye(col);
temp1=diag(diag(Dv).^(-0.5));
temp2=diag(diag(D_SIM).^(-0.5));
reg_Lh=I-temp1*(Path'*LW/De*Path)*temp1;
reg_SIM=I-temp2*SIM*temp2;
L_H_SIM=sigma*reg_Lh+(1-sigma)*reg_SIM;
%%
%高斯核《similarity network fusion for aggregating data types on a genomic scale》
D_Con_SIM = diag(sum(Con_SIM));
L_Con_SIM = D_Con_SIM-Con_SIM;
%%
% Matrix1=H;            %基因通路邻接矩阵
% Matrix2=F_Weight;     %欧氏距离邻接矩阵
% Matrix3=A;            %皮尔逊相关系数邻接矩阵
% Matrix4=B;            %Modularity矩阵
% L1=L;                 %基因通路拉普拉斯矩阵
% L2=Lh;                %超图拉普拉斯矩阵
% % L3=L_SIM;             %本体相似性拉普拉斯矩阵
% L3=L_SIM_temp;        %基因通路+基因本体拉普拉斯矩阵
% % L5=L_N_SIM;           %基因通路拉普拉斯矩阵+基因本体拉普拉斯矩阵
% L4=L_H_SIM;           %超图拉普拉斯矩阵+基因本体拉普拉斯矩阵
% % L7=L_Con_SIM1;        %高斯核《HGIMDA: Heterogeneous graph inference for miRNA-disease association prediction》
% L5=L_Con_SIM;        %高斯核《similarity network fusion for aggregating data types on a genomic scale》
end

