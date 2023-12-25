function [Beta,intercept]=Ada_HRM_Logistics(X,y,Path,A,alpha,lambda,sigma)
[row,col]=size(X);
% [paths,~]=size(Path);
% 
% Path(sum(Path,2)==0,:)=[];

for i = 1 : col
    miu(i) = corr(y,X(:,i));
    if isnan(miu(i)) == 1 || (miu(i) == 0)
        miu(i) = 0.0001;
    end
end
miu1 = 1./(miu.^2);

% A=abs(corr(X));
% A=A-diag(diag(A));
% for i = 1: col
%     for j = 1: col
%         if A(i,j) < 0.0 || isnan(A(i,j))==1
%             A(i,j) = 0 ;
%         end
%     end
% end
% A(find(H==1))=0;

% oe=ones(col,1);
% Weight=Path*oe;
% LW=diag(Weight);
% De=diag(Weight);
% Dv=diag(Weight'*Path);
% Lh=Dv-(Path'*LW/De*Path);

% B=Lh*alpha+A*sigma;

Lh=Path;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
temp = sum(y)/row;
beta_zero = log(temp/(1-temp));    %intercept
beta = zeros(col,1);

iter = 0;
maxiter = 50;
obj=[];
while iter < maxiter %true
        
    beta_temp = beta;
    beta_zero_temp = beta_zero;
        
    eta = beta_zero_temp + X*beta_temp;   %%%%% eta= intercept + X*beta;
    Pi = exp(eta)./(1+exp(eta)) ;      
    W = diag(Pi.*(1-Pi));           %%%%%%%%% W is diagonal matrix%%%%%%%%%%      
    r = (W^-1)*(y-Pi);            %residual= (w^-1)*(y-pi)
   
    %%%%%%%%%%%%%%%%%%%% intercept%%%%%%%%%%%%%%%%%%%%%%   
    beta_zero = sum(y-Pi)/sum(sum(W)); %+ beta_zero_temp;   
    r = r - (beta_zero - beta_zero_temp);    
    for j=1:col
            
        part1 = 0;
        part2=X(:,j)'*W*X(:,j)/row+Lh(j,j)*2*alpha;
            
        for k = 1:col
            if(k==j)
                continue;
            end
%             part1=part1+2*B(k,j)*abs(beta(k));
%             part1=part1+2*alpha*Lh(k,j)*abs(beta(k));
%             part1=part1+2*sigma*A(k,j)*abs(beta(k));
            part1=part1+2*alpha*Lh(k,j)*abs(beta(k))+2*sigma*A(k,j)*abs(beta(k));
        end
            
        r_temp = r + X(:,j)*beta_temp(j);
        S=(X(:,j)'*W*r_temp)/row;
            
        if S > abs(lambda*miu1(j)+part1)
            beta(j) = (S - (lambda*miu1(j)+part1))/part2;%/(1+lambda*(1-alpha));
        elseif S < -abs(lambda*miu1(j)+part1)
            beta(j) = (S + lambda*miu1(j)+part1)/part2;%/(1+lambda*(1-alpha));
        elseif abs(S) <= abs(lambda*miu1(j)+part1)
            beta(j) = 0;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%--update r---%%%%%%%%%%%%%%%%%%%%%%%%%%
        r= r - X(:,j)*(beta(j)-beta_temp(j));
            
    end
%     if norm(beta_temp - beta) < (1E-5)
%         break;
%     end
    
    temp1=beta_zero+X*beta;
    temp2=mean(y.*temp1-log(1+exp(temp1)));
    temp3=lambda*sum(abs(beta).*miu1');
    temp4=alpha*abs(beta')*(Lh)*abs(beta)+sigma*abs(beta')*(A)*abs(beta);
    obj=[obj,temp3+temp4-temp2];
    
    iter = iter + 1;
%     if iter>3&&abs(obj(iter)-obj(iter-1))/abs(obj(iter-1))<(1E-5)
%         break;
%     end
end
save('AWHRM_obj','obj');
Beta = beta;
intercept = beta_zero;  
end

