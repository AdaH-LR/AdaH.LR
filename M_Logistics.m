function [Beta,intercept]=M_Logistics(X,y,Path,alpha,lambda)
[row,col]=size(X);
% [paths,~]=size(Path);
% 
lambda1=lambda;
lambda2=alpha;
% 
% H=zeros(col);
% for i=1:col
%     for j=1:paths
%         if(Path(j,i)==0)
%             continue;
%         end
%         for k=1:col
%             if(Path(j,k)~=0&&i~=k)
%                 H(i,k)=1;
%                 H(k,i)=1;
%             end
%         end
%     end
% end
% H=H-diag(diag(H));
% WL=sum(H);
% L_num=sum(sum(H));
% B=zeros(col);
% for i=1:col
%     for j=1:col
%         if i~=j
%             B(i,j)=H(i,j)-WL(i)*WL(j)/(L_num);
%         end
%     end
% end

B=Path;
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
        
        r_temp = r + X(:,j)*beta_temp(j);
        
        part1 = 0;
        part2=X(:,j)'*W*X(:,j)/row-B(j,j)*2*lambda2;
            
        for k = 1:col
            if(k==j)
                continue;
            end
            part1=part1+B(k,j)*beta(k);
        end

        part1=2*lambda2*part1;
            
        S=(X(:,j)'*W*r_temp)/row +part1;
            
        if S > abs(lambda1)
            beta(j) = (S - (lambda1))/part2;%/(1+lambda*(1-alpha));
        elseif S < -abs(lambda1)
            beta(j) = (S + lambda1)/part2;%/(1+lambda*(1-alpha));
        elseif abs(S) <= abs(lambda1)
            beta(j) = 0;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%--update r---%%%%%%%%%%%%%%%%%%%%%%%%%%
        r= r - X(:,j)*(beta(j)-beta_temp(j));
            
    end
    
%     if norm(beta_temp - beta) < (1E-5)
%         break;
%     end
    
    iter = iter + 1;
    
    temp1=beta_zero+X*beta;
    temp2=mean(y.*temp1-log(1+exp(temp1)));
    temp3=lambda*sum(abs(beta));
    temp4=alpha*beta'*B*beta;
    obj=[obj,temp3-temp4-temp2];
    
    if iter>3&&abs(obj(iter)-obj(iter-1))/abs(obj(iter-1))<(1E-5)
        break;
    end
%     save('M_obj','M_obj');

end
Beta = beta;
intercept = beta_zero;  
end
