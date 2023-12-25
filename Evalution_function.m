function [ YI,F1,AUC,predict_y] = Evalution_function( X_test,y_test,beta_path,intercept_path )
for k = 1:length(intercept_path)
    beta = beta_path(:,k);
    predict_y = Predict_function(beta,intercept_path(k),X_test);
    YI(k) = Youden_index(y_test,predict_y);
    F1(k) = F1_Score(y_test,predict_y);
    %auc(k) = Auc(predict_y,y_test);
    a = predict_y==y_test;
    num=sum(a);
    AUC=num/length(y_test);
end

end

