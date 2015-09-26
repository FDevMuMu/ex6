function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
t = 0;
I_C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';
I_sigma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';% input parameters
initial_errors = 1000000;%set a giant value
FinalPrederrors = initial_errors;
result = zeros(length(I_C)*length(I_sigma),3);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
for i = 1:length(I_C)
	for j = 1:length(I_sigma)
		t = t + 1; % count number
		model= svmTrain(X, y, I_C(i, 1), @(x1, x2) gaussianKernel(x1, x2, I_sigma(j, 1))); 
		predictions = svmPredict(model, Xval);
		prederrors = mean(double(predictions ~= yval));
        
        result(t, :) = [I_C(i, 1), I_sigma(j, 1), prederrors]; 


		
	end
end

sorted_result = sortrows(result, 3);% arrange the result from min to max depends on the 3rd rows 

C = sorted_result(1,1);
sigma = sorted_result(1,2);



      







% =========================================================================

end
