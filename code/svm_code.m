filename = 'documents/psa_data_noinvasion.xlsx';
data = readtable(filename);
load class1
load class2
data1 = data(class1,:);
data2 = data(class2,:);

%Suppose your data is an M by N matrix where N-1 belongs to the features, M is the number of samples/rows (subjects for our case) and the last column is the class.   

%Then we define Feature and Target as: 

 Target=[data1.recurrence;data2.recurrence];

Feature1= [data1.age,data1.cytoreductive,data1.path_T,data1.readmin30,data1.male,data1.los,data1.partialNeph,data1.radicalNeph,data1.lap,data1.robot,data1.open,data1.ebl,data1.ischemia,data1.tumorSize,data1.comorbidTotal,data1.invasion];
Feature2= [data2.age,data2.cytoreductive,data2.path_T,data2.readmin30,data2.male,data2.los,data2.partialNeph,data2.radicalNeph,data2.lap,data2.robot,data2.open,data2.ebl,data2.ischemia,data2.tumorSize,data2.comorbidTotal,data2.invasion];
Feature=[Feature1;Feature2];
%For your case, I'm talking about an Excel sheet after propensity matching that you have the same number of subjects in the two classes. 

%Then you use 
cv = cvpartition(Target, 'HoldOut', 0.3);
%to hold 30% of the data in each class for testing and 70% for training as follow

idxTrain = training(cv);
idxTest = test(cv);
FeatureTrain = Feature(idxTrain, :);
TargetTrain = Target(idxTrain);
FeatureTest = Feature(idxTest, :);
TargetTest = Target(idxTest);
 % Then the following lines define the SVM model based on training data and calculate the training and test results.
  Mdl = fitcsvm(FeatureTrain, TargetTrain, 'Standardize', true, 'KernelFunction', 'RBF', 'KernelScale', 'auto', 'OutlierFraction', 0.05);
        % Extract predicted label for train data
        [TrainSVM, ~] = predict(Mdl, FeatureTrain);
        % Extract predicted label for test data
        [TestSVM, score] = predict(Mdl, FeatureTest);


 II1=find(TargetTest==1);
 II2=find(TargetTest==2);

 I1=find(TargetTrain==1);
 I2=find(TargetTrain==2);


 % Evaluate the performance
 % Confusion Matrix
        FPSVM=length(find(TestSVM(II2)==1)); % find false positives
        TPSVM=length(find(TestSVM(II1)==1)); % find true positives
        TNSVM=length(find(TestSVM(II2)==2)); % find true negatives
        FNSVM=length(find(TestSVM(II1)==2));  % find false negatives

        SSVM=length(find(TestSVM(II1)==1))/length(II1)*100; % Sensitivity
        PrSVM=length(find(TestSVM(II1)==1))/(FPSVM+TPSVM)*100; % Precision
        SPSVM=length(find(TestSVM(II2)==2))/length(II2)*100; %Specifisity
        FPRSVM=length(find(TestSVM(II2)==1))/length(II2)*100; % False positive rate