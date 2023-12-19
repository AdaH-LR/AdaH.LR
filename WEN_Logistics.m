function [Beta,intercept]=WEN_Logistics(X,y,alpha,lambda)
%在通路的基础上考虑了统计学相关性
[row,col]=size(X);

%%
%权值
miu1=[];
for i=1:col
    mean_c1=mean(X(find(y==1),i));
    mean_c2=mean(X(find(y==0),i));
    mean_all=mean(X(:,i));
    BSS=(mean_c1-mean_all).^2+(mean_c2-mean_all).^2;
    WSS=0;
    for j=1:row
        if y(j)==1
            WSS=WSS+(X(j,i)-mean_c1).^2;
        elseif y(j)==0
            WSS=WSS+(X(j,i)-mean_c2).^2;
        end
    end
    miu1(i)=abs(WSS/BSS);
end

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
            
        part2=X(:,j)'*W*X(:,j)/row+2*alpha;
        
        r_temp = r + X(:,j)*beta_temp(j);
        S=(X(:,j)'*W*r_temp)/row;
            
        if S > abs(lambda*miu1(j))
            beta(j) = (S - (lambda*miu1(j)))/part2;%/(1+lambda*(1-alpha));
        elseif S < -abs(lambda*miu1(j))
            beta(j) = (S + lambda*miu1(j))/part2;%/(1+lambda*(1-alpha));
        elseif abs(S) <= abs(lambda*miu1(j))
            beta(j) = 0;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%--update r---%%%%%%%%%%%%%%%%%%%%%%%%%%
        r= r - X(:,j)*(beta(j)-beta_temp(j));
            
    end
    
    temp1=beta_zero+X*beta;
    temp2=mean(y.*temp1-log(1+exp(temp1)));
    temp3=lambda*sum(abs(beta).*miu1');
    temp4=alpha*(beta')*(beta);
    obj=[obj,temp3+temp4-temp2];
    
    iter = iter + 1;
    if iter>3&&abs(obj(iter)-obj(iter-1))/abs(obj(iter-1))<(1E-5)
        break;
    end

end
Beta = beta;
intercept = beta_zero;  
end