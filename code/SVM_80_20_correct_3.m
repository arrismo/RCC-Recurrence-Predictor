clear all;

% Add the necessary directory to the MATLAB path
% addpath(genpath('/Users/marysanchez/Desktop/Suicide Ideation Project /8. Propensity Score/80 Training 20 Testing/NEW/80_20'));

% Define the directory where the R files are located
% directory = ('/Users/marysanchez/Desktop/Suicide Ideation Project /8. Propensity Score/80 Training 20 Testing/NEW/5Fold/Group 3_2');
% files = dir(fullfile(directory,'R-5FoldSI-RA-PSthres22_80_20_Group3_2.mat')); %(From 5Fold second CVP Code)
% nfiles = length(files);

%load R-5FoldSI-RA-PSthres22_80_20_Group1
%load R-SI-RA-PSthres22Group1

load R-SI-RA-PSthres22_80_20_Group1

for q = 1%:nfiles

    % Load train and test datasets
    load FeatureTrain_80_20_RA_Group1.mat; %(From 80_20 first CVP Code)
    load TargetTrain_80_20_RA_Group1.mat; %(From 80_20 first CVP Code)
    load FeatureTest_80_20_RA_Group1.mat; %(From 80_20 first CVP Code)
    load TargetTest_80_20_RA_Group1.mat; %(From 80_20 first CVP Code)

    %FeatureTrain and Target Train 80-20
    Feature = FeatureTrain;
    Target = TargetTrain;

    % Load the R file
    % loadname = files(q).name;
    % fullpath = fullfile(directory,loadname);
    % load(fullpath);

    % Select the top 50 features from the R file
    R = R(1:50);

    for nf = 1:30

        k = nf;

        % Train Support Vector Machine (SVM) classifier
        Mdl = fitcsvm(Feature(:, R(1:nf)), Target, 'Standardize', true, 'KernelFunction', 'RBF', 'KernelScale', 'auto', 'OutlierFraction', 0.05);

        % Extract predicted label for train data
        [TrainSVM, score] = predict(Mdl, Feature(:, R(1:nf)));

        % Extract predicted label for test data
        [TestSVM, score] = predict(Mdl, FeatureTest(:, R(1:nf)));

        II1=find(TargetTest==1);
        II2=find(TargetTest==2);

        I1=find(Target==1);
        I2=find(Target==2);

        % Evaluate the performance (for more information please visit
        % https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
        FPSVM=length(find(TestSVM(II2)==1)); % find false positives
        TPSVM=length(find(TestSVM(II1)==1)); % find true positives
        TNSVM=length(find(TestSVM(II2)==2)); % find true negatives
        FNSVM=length(find(TestSVM(II1)==2));  % find false negatives

        % Matthews correlation coefficient (MCC)
        MCCSVM(1,k)=(TPSVM*TNSVM-FPSVM*FNSVM)/sqrt((TPSVM+FPSVM)*(TPSVM+FNSVM)*(TNSVM+FPSVM)*(TNSVM+FNSVM));


        FPTrSVM=length(find(TrainSVM(I2)==1)); %find false posives for training phase
        TPTrSVM=length(find(TrainSVM(I1)==1));  %find true posives for training phase


        SSVM(1,k)=length(find(TestSVM(II1)==1))/length(II1)*100; % Sensitivity
        PrSVM(1,k)=length(find(TestSVM(II1)==1))/(FPSVM+TPSVM)*100; % Precision
        SPSVM(1,k)=length(find(TestSVM(II2)==2))/length(II2)*100; %Specifisity
        FPRSVM(1,k)=length(find(TestSVM(II2)==1))/length(II2)*100; % False positive rate


        StrSVM(1,k)=length(find(TrainSVM(I1)==1))/length(I1)*100; % Sensitivity of training phase
        PrtrSVM(1,k)=length(find(TrainSVM(I1)==1))/(FPTrSVM+TPTrSVM)*100; %Precision of Training phase
        SPtrSVM(1,k)=length(find(TrainSVM(I2)==2))/length(I2)*100; % Specificity of Training phase

        PSVM(1,k)=100-norm([TestSVM-TargetTest]).^2/length(TargetTest)*100; % Accuracy
        PtrSVM(1,k)=100-norm([TrainSVM-Target]).^2/length(Target)*100; % Accuracy of the training phase

        F1SVM(1,k)= 2*PrSVM(1,k)*SSVM(1,k)/(PrSVM(1,k)+SSVM(1,k)); % F1 score
        F1trSVM(1,k)= 2*PrtrSVM(1,k)*StrSVM(1,k)/(PrtrSVM(1,k)+StrSVM(1,k)); % F1 score of the training phase

        %Calculating Cohen's kappa score for training phase (https://en.wikipedia.org/wiki/Cohen%27s_kappa)
        SVMC1=length(find(TestSVM==1));
        SVMC2=length(find(TestSVM==2));
        L=length(TargetTest);
        ObAC=(length(find(TestSVM(II1)==1))+length(find(TestSVM(II2)==2)))/L;
        ExAC=((length(II1)*SVMC1)/L+(length(II2)*SVMC2)/L)/L;
        KappaSVMT(1,k)=(ObAC-ExAC)/(1-ExAC);

        %Calculating Cohen's kappa score for training phase (https://en.wikipedia.org/wiki/Cohen%27s_kappa)
        SVMTrC1=length(find(TrainSVM==1));
        SVMTrC2=length(find(TrainSVM==2));
        LTr=length(Target);
        ObACTr=(length(find(TrainSVM(I1)==1))+length(find(TrainSVM(I2)==2)))/LTr;
        ExACTr=((length(I1)*SVMTrC1)/LTr+(length(I2)*SVMTrC2)/LTr)/LTr;
        KappaSVMTr(1,k)=(ObACTr-ExACTr)/(1-ExACTr);

        %% Calculate classification accuracy for different number of features
        for i=nf
            Per(i,:)=[i mean(PSVM(:,i))]; %Test accuracy
            PerT(i,:)=[i mean(PtrSVM(:,i))]; % training accuracy
        end


        % TESTING: Calculate evaluation parameters for different number of features
        for mr=nf
            Kappa(mr,:)=[mr mean(KappaSVMT(:,mr)) ];
            F1(mr,:)=[mr mean(F1SVM(:,mr)) ];
            PPV(mr,:)=[mr mean(PrSVM(:,mr)) ];
            SE(mr,:)=[mr mean(SSVM(:,mr)) ];
            SP(mr,:)=[mr mean(SPSVM(:,mr)) ];
            FPR(mr,:)=[mr mean(FPRSVM(:,mr))];
            MCC(mr,:)=[mr mean(MCCSVM(:,mr))];
        end


        I=find(Per(:,2)==max(Per(:,2)));
        I=I(1);
        Performance_Test=[I SE(I,2) SP(I,2) Per(I,2) F1(I,2) Kappa(I,2) FPR(I,2) PPV(I,2) MCC(I,2)];

        clear Sen Spe TA F1A Prec MCCM KappaM


        Performance_Test;
        Sen=Performance_Test(:,2);
        Spe=Performance_Test(:,3);
        TA=Performance_Test(:,4);
        F1A=Performance_Test(:,5);
        KappaM=Performance_Test(:,6);
        Prec=Performance_Test(:,8);
        MCCM=Performance_Test(:,9);
        


        II=find(isnan(Prec) | isnan(F1A) | isnan(MCCM) | isnan(KappaM));
        Sen(II)=[];
        Spe(II)=[];
        TA(II)=[];
        F1A(II)=[];
        Prec(II)=[];
        MCCM(II)=[];
        KappaM(II)=[];

        %%
        MeanPer(nf,:)=[mean(Sen), mean(Spe), mean(TA), mean(Prec), mean(F1A), mean(MCCM), mean(KappaM)]; %averaged performance over 100 runs for each number of features from 1 to all 50 features
        StdPerf(nf,:)=[std(Sen), std(Spe), std(TA), std(Prec), std(F1A), std(MCCM), std(KappaM)]; %standard deviation of the performance over 100 runs for each number of features from 1 to all 50 features
        Ind{nf}=I; % tells you the index of the runs with NaN values at each nf


        % Save results
        savesuff = 'SI-PS-RA-thres22-Group1'; %Suicide Ideation
        savenameR = strcat('R-SVM-80-20',savesuff);
        savenameFTrain = strcat('TrainSVM-SVM-80-20',savesuff);
        savenameTTrain = strcat('TestSVM-SVM-80-20',savesuff);
        saveMean = strcat('MeanPerf-SVMTest', savesuff);
        saveStd = strcat('StdPerf-SVMTest', savesuff);

        save(savenameR, 'R');
        save(savenameFTrain, 'TrainSVM');
        save(savenameTTrain, 'TestSVM');
        save(saveMean, 'MeanPer');
        save(saveStd, 'StdPerf');


        %% TRAINING Performance

        for mr=nf
            Kappa_Tr(mr,:)=[mr mean(KappaSVMTr(:,mr)) ];
            F1_Tr(mr,:)=[mr mean(F1trSVM(:,mr)) ];
            PPV_Tr(mr,:)=[mr mean(PtrSVM(:,mr)) ];
            SE_Tr(mr,:)=[mr mean(StrSVM(:,mr)) ];
            SP_Tr(mr,:)=[mr mean(SPtrSVM(:,mr)) ];
            %FPR_Tr(mr,:)=[mr mean(PrtrSVM(:,mr))];
            %MCC_Tr(mr,:)=[mr mean(MCCSVM(:,mr))]; %Don't KNOW
        end

        I=find(PerT(:,2)==max(PerT(:,2)));
        I=I(1);
        Performance_Train=[I SE_Tr(I,2) SP_Tr(I,2) PerT(I,2) F1_Tr(I,2) Kappa_Tr(I,2) FPR(I,2) PPV(I,2)]; %MCC(I,2)];

        clear Sen Spe TA F1A Prec MCCM KappaM


        Performance_Train;
        Sen_Tr=Performance_Train(:,2);
        Spe_Tr=Performance_Train(:,3);
        TA_Tr=Performance_Train(:,4);
        F1A_Tr=Performance_Train(:,5);
        KappaM_Tr=Performance_Train(:,6);
        Prec_Tr=Performance_Train(:,8);
        %MCCM_Tr=Performance_Train(:,9);



        II=find(isnan(Prec_Tr) | isnan(F1A_Tr) | isnan(KappaM_Tr)); %| isnan(MCCM_Tr) |
        Sen_Tr(II)=[];
        Spe_Tr(II)=[];
        TA_Tr(II)=[];
        F1A_Tr(II)=[];
        Prec_Tr(II)=[];
        %MCCM_Tr(II)=[];
        KappaM_Tr(II)=[];

        %%
        MeanPer_Tr(nf,:)=[mean(Sen_Tr), mean(Spe_Tr), mean(TA_Tr), mean(Prec_Tr), mean(F1A_Tr), mean(KappaM_Tr)]; %, mean(MCCM_Tr); %averaged performance over 100 runs for each number of features from 1 to all 50 features
        StdPerf_Tr(nf,:)=[std(Sen_Tr), std(Spe_Tr), std(TA_Tr), std(Prec_Tr), std(F1A_Tr), std(KappaM_Tr)]; %, std(MCCM_Tr); %standard deviation of the performance over 100 runs for each number of features from 1 to all 50 features
        Ind{nf}=I; % tells you the index of the runs with NaN values at each nf


        % Save results
        saveMean = strcat('MeanPerf-SVMTrain', savesuff);
        saveStd = strcat('StdPerf-SVMTrain', savesuff);

        save(saveMean, 'MeanPer_Tr');
        save(saveStd, 'StdPerf_Tr');

    end
end
