clear all
filename = 'psa_data_noinvasion.xlsx';
data = readtable(filename);
% load class1
% load class2
% data1 = data(class1,:);
% data2 = data(class2,:);

data1 = data(1:321,:);
data2 = data(322:642,:);

column_names = data1.Properties.VariableNames;

for i = 1:numel(column_names)
    column_name = column_names{i};
    
    
    if iscell(data1.(column_name))
        data1.(column_name) = cellfun(@str2double, data1.(column_name));
    end
end


column_names = data2.Properties.VariableNames;

for i = 1:numel(column_names)
    column_name = column_names{i};
    
    
    if iscell(data2.(column_name))
        data2.(column_name) = cellfun(@str2double, data2.(column_name));
    end
end



%Suppose your data is an M by N matrix where N-1 belongs to the features, M is the number of samples/rows (subjects for our case) and the last column is the class.   


%Then we define Feature and Target as: 

Target=[data1.recurrence;data2.recurrence];




Feature1= [data1.age,data1.cytoreductive,data1.path_T,data1.readmin30,data1.male,data1.los,data1.partialNeph,data1.radicalNeph,data1.lap,data1.robot,data1.open,data1.ebl,data1.ischemia,data1.tumorSize,data1.comorbidTotal,data1.invasion];
Feature2= [data2.age,data2.cytoreductive,data2.path_T,data2.readmin30,data2.male,data2.los,data2.partialNeph,data2.radicalNeph,data2.lap,data2.robot,data2.open,data2.ebl,data2.ischemia,data2.tumorSize,data2.comorbidTotal,data2.invasion];
Feature=[Feature1;Feature2];

%Then you use 
cv = cvpartition(Target, 'HoldOut', 0.2);
%to hold 20% of the data in each class for testing and 80% for training as follow

%cv = cvpartition(Target,"KFold",5);

idxTrain = training(cv,"all");
idxTest = test(cv,"all");
FeatureTrain = Feature(idxTrain);
TargetTrain = Target(idxTrain);
FeatureTest = Feature(idxTest);
TargetTest = Target(idxTest);

 % Then the following lines define the SVM model based on training data and calculate the training and test results.
   Mdl = fitcsvm(Feature, Target, 'Standardize', true, 'KernelFunction', 'RBF', 'KernelScale', 'auto', 'OutlierFraction', 0.05);
   
   cv = crossval(Mdl, 'KFold',5);
   % now we can see the cross validation loss over 5 folds 
fprintf('Average cross-validation loss: %.2f\n', loss);


         % Extract predicted label for train data
         [TrainSVM, score] = predict(Mdl, FeatureTrain);
         % Extract predicted label for test data
         [TestSVM, score] = predict(Mdl, FeatureTest);


% create confusion matrix
cm = confusionchart(TargetTrain,TrainSVM)
     



 II1=find(TargetTest==0);
 II2=find(TargetTest==1);

 I1=find(TargetTrain==0);
 I2=find(TargetTrain==1);


 % Evaluate the performance


        PSVM=100-norm(TestSVM-TargetTest).^2/length(TargetTest)*100; % Accuracy
        PtrSVM=100-norm(TrainSVM-TargetTrain).^2/length(TargetTrain)*100; % Accuracy of the training phase
        % 
        % F1SVM= 2*PtrSVM*SSVM/(t+); % F1 score
        % F1trSVM= 2*PrtrSVM*StrSVM/(PrtrSVM+StrSVM); % F1 score of the training phase

 % Confusion Matrix
        FPSVM=length(find(TestSVM(II2)==0)); % find false positives
        TPSVM=length(find(TestSVM(II1)==0)); % find true positives 
        TNSVM=length(find(TestSVM(II2)==1)); % find true negatives
        FNSVM=length(find(TestSVM(II1)==1));  % find false negatives

        SSVM=length(find(TestSVM(II1)==0))/length(II1)*100; % Sensitivity
        PrSVM=length(find(TestSVM(II1)==0))/(FPSVM+TPSVM)*100; % Precision
        SPSVM=length(find(TestSVM(II2)==1))/length(II2)*100; %Specifisity
        FPRSVM=length(find(TestSVM(II2)==1))/length(II2)*100; % False positive rate