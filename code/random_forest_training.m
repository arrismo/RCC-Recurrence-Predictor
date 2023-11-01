% Random Forest Model


clear all
filename = 'psa_data_noinvasion.xlsx';
data = readtable(filename);
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

Target=[data1.recurrence;data2.recurrence];




Feature1= [data1.age,data1.cytoreductive,data1.path_T,data1.readmin30,data1.male,data1.los,data1.partialNeph,data1.radicalNeph,data1.lap,data1.robot,data1.open,data1.ebl,data1.ischemia,data1.tumorSize,data1.comorbidTotal,data1.invasion];
Feature2= [data2.age,data2.cytoreductive,data2.path_T,data2.readmin30,data2.male,data2.los,data2.partialNeph,data2.radicalNeph,data2.lap,data2.robot,data2.open,data2.ebl,data2.ischemia,data2.tumorSize,data2.comorbidTotal,data2.invasion];
Feature=[Feature1;Feature2];



cv = cvpartition(Target, 'HoldOut', 0.2);



idxTrain = training(cv);
idxTest = test(cv);
FeatureTrain = Feature(idxTrain, :);
TargetTrain = Target(idxTrain);
FeatureTest = Feature(idxTest, :);
TargetTest = Target(idxTest);

 % Random Forest 
 numTrees = 100;
 rMdl = TreeBagger(numTrees,FeatureTrain, TargetTrain, 'Method','classification');
 [TrainRF, score] = predict(rMdl, FeatureTrain);
 [TestRF, score] = predict(rMdl, FeatureTest);


% this is wrong
cm = confusionchart(TargetTrain,TrainRF);


 II1=find(TargetTest==0);
 II2=find(TargetTest==1);

 I1=find(TargetTrain==0);
 I2=find(TargetTrain==1);


 % Evaluate the performance


        PSRF=100-norm(TestRF-TargetTest).^2/length(TargetTest)*100; % Accuracy
        PtrRF=100-norm(TrainRF-TargetTrain).^2/length(TargetTrain)*100; % Accuracy of the training phase
        % 
        % F1SVM= 2*PtrSVM*SSVM/(t+); % F1 score
        % F1trSVM= 2*PrtrSVM*StrSVM/(PrtrSVM+StrSVM); % F1 score of the training phase
