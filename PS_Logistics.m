function [Beta,intercept]=PS_Logistics(X,y,alpha,lambda)
[row,col] = size(X);
Weight = abs(corr(X)).^4;
for i = 1: col
    for j = 1: col
        if Weight(i,j) < 0.3 || isnan(Weight(i,j))==1
            Weight(i,j) = 0 ;
        end
    end
end
%%
%È¨Öµ
for i = 1 : col
    miu(i) = corr(y,X(:,i));
    if isnan(miu(i)) == 1 || (miu(i) == 0)
        miu(i) = 0.0001;
    end
end
miu1 = 1./(miu.^2);
%%

    
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
        part2 = 1;
            
        for k = 1:col
            part1 =  part1 + Weight(k,j)* miu(k)* miu(j)*(-beta(k));
            part2 =  part2 + Weight(k,j)* miu(j)* miu(j);
        end
            
        part1 =  part1 -  miu(j)* miu(j)*(-beta(j));
        part2 =  part2 -  miu(j)* miu(j);
        part1 =  part1 * lambda * 2;
        part2 =  part2 * alpha * 2 + 1;
            
        S=(X(:,j)'*W*r)/row + beta_temp(j) + part1;
            
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
%     if norm(beta_temp - beta) < (1E-5)
%         break;
%     end
    
    temp1=beta_zero+X*beta;
    temp2=mean(y.*temp1-log(1+exp(temp1)));
    temp3=lambda*sum(abs(beta).*miu1');
    temp4=0;
    for i=1:col
        for j=1:col
            if i==j
                continue;
            end
            temp4=temp4+Weight(i,j)*(miu(i)*beta(i)-miu(j)*beta(j))^2;
        end
    end
    temp4=temp4*alpha;
    obj=[obj,temp3+temp4-temp2];
    
    iter = iter + 1;
    if iter>3&&abs(obj(iter)-obj(iter-1))/abs(obj(iter-1))<(1E-5)
        break;
    end
%     save('AL_obj','AL_obj');
end
Beta = beta;
intercept = beta_zero;  
end
