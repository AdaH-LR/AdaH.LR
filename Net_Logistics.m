function [Beta,intercept]=Net_Logistics(X,y,Path,alpha,lambda)
[row,col]=size(X);
% [paths,~]=size(Path);
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
% oe=ones(col,1);
% D=diag(H*oe);
% L=D-H;
% % I=eye(col);
% % L=I-D.^(-0.5)*H*D.^(-0.5);

L=Path;
    
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
        part2=X(:,j)'*W*X(:,j)/row+L(j,j)*2*alpha;
            
        for k = 1:col
            if(k==j)
                continue;
            end
            part1=part1+L(k,j)*(beta(k));
        end

        part1=2*alpha*part1;
            
        S=(X(:,j)'*W*r_temp)/row - part1;
            
        if S > (lambda)
            beta(j) = (S - (lambda))/part2;%/(1+lambda*(1-alpha));
        elseif S < -(lambda)
            beta(j) = (S + lambda)/part2;%/(1+lambda*(1-alpha));
        elseif abs(S) <= (lambda)
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
    temp4=alpha*beta'*L*beta;
    obj=[obj,temp3+temp4-temp2];
    
    if iter>3&&abs(obj(iter)-obj(iter-1))/abs(obj(iter-1))<(1E-5)
        break;
    end
%     save('L_obj','L_obj');
end
Beta = beta;
intercept = beta_zero;  
end
