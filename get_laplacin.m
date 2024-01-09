function [H,F_Weight,A,B,L,Lh,L_SIM_temp,L_H_SIM,L_Con_SIM]=get_laplacin(X,Path,SIM,Con_SIM)
[row,col]=size(X);
oe=ones(col,1);

Path(sum(Path,2)==0,:)=[];
% Empty_Index=find(sum(Path,2)==0);

[paths,~]=size(Path);
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
DL=diag(H*oe);
L=DL-H;
%%
Weight=Path*oe;
LW=diag(Weight);
De=diag(Weight);
Dv=diag(Weight'*Path);
Lh=Dv-(Path'*LW/De*Path);
%%
%modularity
WL=sum(H);  
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
D_SIM = diag(sum(SIM));
L_SIM = D_SIM-SIM;
%%
% A = (1-sigma)*SIM+sigma*F_Weight;
% A(find(H==0)) = 0;
% D2 = diag(sum(A));
% L2 = D2-A;
%%
SIM_temp = SIM;
SIM_temp(find(H==0))=0;
D_SIM_temp = diag(sum(SIM_temp));
L_SIM_temp = D_SIM_temp-SIM_temp;
%%
sigma=0.5;
I=eye(col);
temp1=diag(diag(Dv).^(-0.5));
temp2=diag(diag(D_SIM).^(-0.5));
reg_Lh=I-temp1*(Path'*LW/De*Path)*temp1;
reg_SIM=I-temp2*SIM*temp2;
L_H_SIM=sigma*reg_Lh+(1-sigma)*reg_SIM;
%%
%《similarity network fusion for aggregating data types on a genomic scale》
D_Con_SIM = diag(sum(Con_SIM));
L_Con_SIM = D_Con_SIM-Con_SIM;
% Matrix1=H;            
% Matrix2=F_Weight;     
% Matrix3=A;            
% Matrix4=B;           
% L1=L;                
% L2=Lh;                
% % L3=L_SIM;             
% L3=L_SIM_temp;       
% % L5=L_N_SIM;         
% L4=L_H_SIM;          
% % L7=L_Con_SIM1;        %《HGIMDA: Heterogeneous graph inference for miRNA-disease association prediction》
% L5=L_Con_SIM;        %《similarity network fusion for aggregating data types on a genomic scale》
end

