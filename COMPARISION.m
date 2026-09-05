%% ================================================================
% SMART-GRID SHIELD
% ML ALGORITHM COMPARISON
%
% Algorithms:
% 1. Decision Tree
% 2. KNN
% 3. SVM
% 4. Random Forest
%
% Metrics:
% Accuracy
% Precision
% Recall
% F1 Score
%
% Dataset:
% Smart_Grid_Shield_Realistic_Overlapping_85_90_Dataset-3.xlsx
% ================================================================

clc;
clear;
close all;

fprintf('\n');
fprintf('============================================================\n');
fprintf('       SMART-GRID SHIELD ML ALGORITHM COMPARISON\n');
fprintf('============================================================\n\n');


%% ================================================================
% 1. LOAD DATASET
% ================================================================

filename = 'Smart_Grid_Shield_Realistic_Overlapping_85_90_Dataset-3.xlsx';

fprintf('Loading dataset...\n');

if ~isfile(filename)
    error(['Dataset file not found: ', filename]);
end

T = readtable(filename);

fprintf('Dataset loaded successfully.\n');
fprintf('Rows    : %d\n', height(T));
fprintf('Columns : %d\n\n', width(T));


%% ================================================================
% 2. DISPLAY COLUMN NAMES
% ================================================================

fprintf('Available columns:\n');

for i = 1:width(T)
    fprintf('%2d. %s\n', i, T.Properties.VariableNames{i});
end

fprintf('\n');


%% ================================================================
% 3. TARGET COLUMN
% ================================================================

targetColumn = 'Condition';

if ~ismember(targetColumn, T.Properties.VariableNames)

    if ismember('Status', T.Properties.VariableNames)
        targetColumn = 'Status';
    else
        error('Target column "Condition" or "Status" not found.');
    end

end

fprintf('Target column: %s\n\n', targetColumn);


%% ================================================================
% 4. FIND AMBIENT TEMPERATURE COLUMN
% ================================================================

possibleAmbientNames = { ...
    'Ambient_Temp_C', ...
    'Ambient_Temperature', ...
    'AmbientTemp_C', ...
    'AmbientTemp', ...
    'Ambient_Temp'};

ambientColumn = '';

for i = 1:length(possibleAmbientNames)

    if ismember(possibleAmbientNames{i}, T.Properties.VariableNames)

        ambientColumn = possibleAmbientNames{i};
        break;

    end

end

if isempty(ambientColumn)

    error('Ambient temperature column not found.');

end

fprintf('Ambient temperature column: %s\n\n', ambientColumn);


%% ================================================================
% 5. FEATURES
% ================================================================

featureNames = { ...
    'PV_Temp_C', ...
    ambientColumn, ...
    'DeltaT_C', ...
    'LDR_ADC', ...
    'PV_Voltage_V'};

fprintf('Features used:\n');

for i = 1:length(featureNames)

    fprintf('%d. %s\n', ...
        i, ...
        featureNames{i});

end

fprintf('\n');


%% ================================================================
% 6. CHECK FEATURE COLUMNS
% ================================================================

for i = 1:length(featureNames)

    if ~ismember(featureNames{i}, T.Properties.VariableNames)

        error(['Feature column not found: ', ...
            featureNames{i}]);

    end

end


%% ================================================================
% 7. EXTRACT FEATURES AND TARGET
% ================================================================

X = T{:, featureNames};

Y = categorical(T.(targetColumn));


%% ================================================================
% 8. REMOVE INVALID VALUES ONLY
%
% Sensor_Anomaly remains a valid class.
% ================================================================

validRows = ...
    all(isfinite(X), 2) & ...
    ~isundefined(Y);

X = X(validRows,:);
Y = Y(validRows);

fprintf('Valid samples: %d\n\n', size(X,1));


%% ================================================================
% 9. CLASS DISTRIBUTION
% ================================================================

fprintf('Class distribution:\n');

classNames = categories(Y);

for i = 1:length(classNames)

    count = sum(Y == classNames{i});

    fprintf('%-25s : %d\n', ...
        classNames{i}, ...
        count);

end

fprintf('\n');


%% ================================================================
% 10. 80/20 TRAIN-TEST SPLIT
% ================================================================

rng(1);

fprintf('Creating 80/20 train-test split...\n');

cvHoldout = cvpartition(Y, 'HoldOut', 0.20);

trainIndex = training(cvHoldout);
testIndex  = test(cvHoldout);

XTrain = X(trainIndex,:);
YTrain = Y(trainIndex);

XTest = X(testIndex,:);
YTest = Y(testIndex);

fprintf('Training samples : %d\n', size(XTrain,1));
fprintf('Testing samples  : %d\n\n', size(XTest,1));


%% ================================================================
% 11. STANDARDIZATION
%
% Mean and standard deviation are calculated ONLY from training
% data to avoid data leakage.
% ================================================================

mu = mean(XTrain, 1);

sigma = std(XTrain, 0, 1);

sigma(sigma == 0) = 1;

XTrainStd = ...
    (XTrain - mu) ./ sigma;

XTestStd = ...
    (XTest - mu) ./ sigma;


%% ================================================================
% 12. DECISION TREE
% ================================================================

fprintf('Training Decision Tree...\n');

treeModel = fitctree( ...
    XTrain, ...
    YTrain, ...
    'MaxNumSplits', 12, ...
    'MinLeafSize', 5);

treeTrainPred = ...
    predict(treeModel, XTrain);

treeTestPred = ...
    predict(treeModel, XTest);

treeTrainAccuracy = ...
    mean(treeTrainPred == YTrain) * 100;

treeTestAccuracy = ...
    mean(treeTestPred == YTest) * 100;

fprintf('Decision Tree Training Accuracy : %.2f%%\n', ...
    treeTrainAccuracy);

fprintf('Decision Tree Testing Accuracy  : %.2f%%\n\n', ...
    treeTestAccuracy);


%% ================================================================
% 13. KNN
% ================================================================

fprintf('Training KNN...\n');

knnModel = fitcknn( ...
    XTrainStd, ...
    YTrain, ...
    'NumNeighbors', 5, ...
    'Distance', 'euclidean', ...
    'Standardize', false);

knnTrainPred = ...
    predict(knnModel, XTrainStd);

knnTestPred = ...
    predict(knnModel, XTestStd);

knnTrainAccuracy = ...
    mean(knnTrainPred == YTrain) * 100;

knnTestAccuracy = ...
    mean(knnTestPred == YTest) * 100;

fprintf('KNN Training Accuracy : %.2f%%\n', ...
    knnTrainAccuracy);

fprintf('KNN Testing Accuracy  : %.2f%%\n\n', ...
    knnTestAccuracy);


%% ================================================================
% 14. SVM
% ================================================================

fprintf('Training SVM...\n');

svmTemplate = templateSVM( ...
    'KernelFunction', 'linear', ...
    'Standardize', false);

svmModel = fitcecoc( ...
    XTrainStd, ...
    YTrain, ...
    'Learners', svmTemplate);

svmTrainPred = ...
    predict(svmModel, XTrainStd);

svmTestPred = ...
    predict(svmModel, XTestStd);

svmTrainAccuracy = ...
    mean(svmTrainPred == YTrain) * 100;

svmTestAccuracy = ...
    mean(svmTestPred == YTest) * 100;

fprintf('SVM Training Accuracy : %.2f%%\n', ...
    svmTrainAccuracy);

fprintf('SVM Testing Accuracy  : %.2f%%\n\n', ...
    svmTestAccuracy);


%% ================================================================
% 15. RANDOM FOREST
% ================================================================

fprintf('Training Random Forest...\n');

rfModel = fitcensemble( ...
    XTrain, ...
    YTrain, ...
    'Method', 'Bag', ...
    'NumLearningCycles', 50);

rfTrainPred = ...
    predict(rfModel, XTrain);

rfTestPred = ...
    predict(rfModel, XTest);

rfTrainAccuracy = ...
    mean(rfTrainPred == YTrain) * 100;

rfTestAccuracy = ...
    mean(rfTestPred == YTest) * 100;

fprintf('Random Forest Training Accuracy : %.2f%%\n', ...
    rfTrainAccuracy);

fprintf('Random Forest Testing Accuracy  : %.2f%%\n\n', ...
    rfTestAccuracy);


%% ================================================================
% 16. STORE ACCURACY RESULTS
% ================================================================

modelNames = { ...
    'Decision Tree'; ...
    'KNN'; ...
    'SVM'; ...
    'Random Forest'};

trainingAccuracy = [ ...
    treeTrainAccuracy; ...
    knnTrainAccuracy; ...
    svmTrainAccuracy; ...
    rfTrainAccuracy];

testingAccuracy = [ ...
    treeTestAccuracy; ...
    knnTestAccuracy; ...
    svmTestAccuracy; ...
    rfTestAccuracy];


%% ================================================================
% 17. PRECISION, RECALL AND F1 SCORE
% ================================================================

[treePrecision, treeRecall, treeF1] = ...
    calculateMetrics( ...
        YTest, ...
        treeTestPred);

[knnPrecision, knnRecall, knnF1] = ...
    calculateMetrics( ...
        YTest, ...
        knnTestPred);

[svmPrecision, svmRecall, svmF1] = ...
    calculateMetrics( ...
        YTest, ...
        svmTestPred);

[rfPrecision, rfRecall, rfF1] = ...
    calculateMetrics( ...
        YTest, ...
        rfTestPred);


precisionValues = [ ...
    treePrecision; ...
    knnPrecision; ...
    svmPrecision; ...
    rfPrecision];

recallValues = [ ...
    treeRecall; ...
    knnRecall; ...
    svmRecall; ...
    rfRecall];

f1Values = [ ...
    treeF1; ...
    knnF1; ...
    svmF1; ...
    rfF1];


%% ================================================================
% 18. PERFORMANCE TABLE IN COMMAND WINDOW
% ================================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('                 MODEL PERFORMANCE\n');
fprintf('============================================================\n');

fprintf( ...
    '%-20s %11s %11s %11s %11s %11s\n', ...
    'Algorithm', ...
    'Accuracy', ...
    'Precision', ...
    'Recall', ...
    'F1', ...
    'Train Acc.');

fprintf('------------------------------------------------------------\n');

for i = 1:length(modelNames)

    fprintf( ...
        '%-20s %10.2f%% %10.2f%% %10.2f%% %10.2f%% %10.2f%%\n', ...
        modelNames{i}, ...
        testingAccuracy(i), ...
        precisionValues(i), ...
        recallValues(i), ...
        f1Values(i), ...
        trainingAccuracy(i));

end

fprintf('============================================================\n\n');


%% ================================================================
% 19. DECISION TREE CONFUSION MATRIX
% ================================================================

figure('Name','Decision Tree Confusion Matrix');

confusionchart( ...
    YTest, ...
    treeTestPred);

title('Decision Tree Confusion Matrix');


%% ================================================================
% 20. KNN CONFUSION MATRIX
% ================================================================

figure('Name','KNN Confusion Matrix');

confusionchart( ...
    YTest, ...
    knnTestPred);

title('KNN Confusion Matrix');


%% ================================================================
% 21. SVM CONFUSION MATRIX
% ================================================================

figure('Name','SVM Confusion Matrix');

confusionchart( ...
    YTest, ...
    svmTestPred);

title('SVM Confusion Matrix');


%% ================================================================
% 22. RANDOM FOREST CONFUSION MATRIX
% ================================================================

figure('Name','Random Forest Confusion Matrix');

confusionchart( ...
    YTest, ...
    rfTestPred);

title('Random Forest Confusion Matrix');


%% ================================================================
% 23. TESTING ACCURACY GRAPH
% ================================================================

figure('Name','Testing Accuracy Comparison');

b = bar(testingAccuracy);

ylabel('Testing Accuracy (%)');

xlabel('Machine Learning Algorithm');

title('ML Algorithm Testing Accuracy');

set(gca, ...
    'XTick', 1:length(modelNames), ...
    'XTickLabel', modelNames);

ylim([0 105]);

grid on;


% Percentage inside each bar

for i = 1:length(testingAccuracy)

    value = testingAccuracy(i);

    text( ...
        i, ...
        value / 2, ...
        sprintf('%.1f%%', value), ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'middle', ...
        'FontWeight', 'bold', ...
        'FontSize', 10, ...
        'Color', 'white');

end


%% ================================================================
% 24. TRAINING VS TESTING ACCURACY
% ================================================================

figure('Name','Training vs Testing Accuracy');

accuracyData = [ ...
    trainingAccuracy, ...
    testingAccuracy];

b = bar(accuracyData, 'grouped');

ylabel('Accuracy (%)');

xlabel('Machine Learning Algorithm');

title('Training vs Testing Accuracy');

set(gca, ...
    'XTick', 1:length(modelNames), ...
    'XTickLabel', modelNames);

ylim([0 105]);

grid on;

legend( ...
    {'Training Accuracy','Testing Accuracy'}, ...
    'Location', 'southoutside', ...
    'Orientation', 'horizontal');


% Percentage inside every bar

for j = 1:size(accuracyData,2)

    xPos = b(j).XEndPoints;

    for i = 1:length(modelNames)

        value = accuracyData(i,j);

        text( ...
            xPos(i), ...
            value / 2, ...
            sprintf('%.1f%%', value), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', ...
            'FontWeight', 'bold', ...
            'FontSize', 8, ...
            'Color', 'white', ...
            'Rotation', 90);

    end

end


%% ================================================================
% 25. FINAL PERFORMANCE GRAPH
%
% Accuracy + Precision + Recall + F1
%
% ALL FOUR METRICS ARE SHOWN FOR EVERY ALGORITHM
% ================================================================

figure('Name','Accuracy Precision Recall F1 Comparison');

performanceData = [ ...
    testingAccuracy(:), ...
    precisionValues(:), ...
    recallValues(:), ...
    f1Values(:)];


% Create vertical grouped bars

b = bar( ...
    performanceData, ...
    'grouped');


%% Axis labels

ylabel('Performance (%)');

xlabel('Machine Learning Algorithm');

title('ML Algorithm Performance Comparison');


%% X-axis

set(gca, ...
    'XTick', 1:length(modelNames), ...
    'XTickLabel', modelNames);


%% Y-axis

ylim([0 105]);


%% Grid

grid on;


%% Legend

legend( ...
    {'Accuracy','Precision','Recall','F1 Score'}, ...
    'Location', 'southoutside', ...
    'Orientation', 'horizontal');


%% ================================================================
% PERCENTAGE INSIDE EVERY INDIVIDUAL BAR
% ================================================================

for j = 1:size(performanceData,2)

    % Get actual X position of each bar
    xPos = b(j).XEndPoints;

    for i = 1:length(modelNames)

        value = performanceData(i,j);

        % Vertical center of the bar
        yPos = value / 2;


        % Display percentage

        text( ...
            xPos(i), ...
            yPos, ...
            sprintf('%.1f%%', value), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', ...
            'FontWeight', 'bold', ...
            'FontSize', 8, ...
            'Color', 'white', ...
            'Rotation', 90);

    end

end


%% ================================================================
% 26. FEATURE IMPORTANCE
% ================================================================

fprintf('Calculating feature importance...\n');

importanceValues = ...
    predictorImportance(treeModel);


figure('Name','Decision Tree Feature Importance');

bar(importanceValues);

set(gca, ...
    'XTick', 1:length(featureNames), ...
    'XTickLabel', featureNames);

xtickangle(30);

ylabel('Importance');

xlabel('Feature');

title('Decision Tree Feature Importance');

grid on;


%% ================================================================
% 27. FEATURE IMPORTANCE VALUES
% ================================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('             FEATURE IMPORTANCE\n');
fprintf('============================================================\n');

for i = 1:length(featureNames)

    fprintf( ...
        '%-25s : %.6f\n', ...
        featureNames{i}, ...
        importanceValues(i));

end

fprintf('============================================================\n\n');


%% ================================================================
% 28. MODEL COMPLEXITY
% ================================================================

treeNodes = ...
    treeModel.NumNodes;

knnSamples = ...
    size(XTrain,1);


% SVM support vectors

svmSupportVectors = 0;

for k = 1:length(svmModel.BinaryLearners)

    svmSupportVectors = ...
        svmSupportVectors + ...
        size( ...
            svmModel.BinaryLearners{k}.SupportVectors, ...
            1);

end


% Random Forest number of trees

rfTrees = ...
    rfModel.NumTrained;


fprintf('============================================================\n');
fprintf('                 MODEL COMPLEXITY\n');
fprintf('============================================================\n');

fprintf( ...
    'Decision Tree nodes        : %d\n', ...
    treeNodes);

fprintf( ...
    'KNN training samples       : %d\n', ...
    knnSamples);

fprintf( ...
    'SVM support vectors        : %d\n', ...
    svmSupportVectors);

fprintf( ...
    'Random Forest trees        : %d\n', ...
    rfTrees);

fprintf('============================================================\n\n');


%% ================================================================
% 29. MODEL COMPLEXITY TABLE
% ================================================================

ComplexityTable = table( ...
    {'Decision Tree'; ...
     'KNN'; ...
     'SVM'; ...
     'Random Forest'}, ...
    [ ...
     treeNodes; ...
     knnSamples; ...
     svmSupportVectors; ...
     rfTrees], ...
    'VariableNames', ...
    {'Algorithm','Complexity_Value'});


%% ================================================================
% 30. PERFORMANCE RESULTS TABLE
% ================================================================

ResultsTable = table( ...
    modelNames, ...
    trainingAccuracy, ...
    testingAccuracy, ...
    precisionValues, ...
    recallValues, ...
    f1Values, ...
    'VariableNames', ...
    {'Algorithm', ...
     'Training_Accuracy_Percent', ...
     'Testing_Accuracy_Percent', ...
     'Precision_Percent', ...
     'Recall_Percent', ...
     'F1_Percent'});


%% ================================================================
% 31. FIVE-FOLD CROSS VALIDATION
% ================================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('             5-FOLD CROSS VALIDATION\n');
fprintf('============================================================\n');

rng(1);

cv5 = cvpartition(Y, 'KFold', 5);

cvAccuracyDT = zeros(5,1);

cvAccuracyKNN = zeros(5,1);

cvAccuracySVM = zeros(5,1);

cvAccuracyRF = zeros(5,1);


for fold = 1:5

    fprintf( ...
        'Processing fold %d/5...\n', ...
        fold);


    %% Training / testing indices

    trainIdx = ...
        training(cv5, fold);

    testIdx = ...
        test(cv5, fold);


    Xtr = X(trainIdx,:);

    Ytr = Y(trainIdx);

    Xte = X(testIdx,:);

    Yte = Y(testIdx);


    %% Standardization

    muCV = ...
        mean(Xtr,1);

    sigmaCV = ...
        std(Xtr,0,1);

    sigmaCV(sigmaCV == 0) = 1;

    XtrStd = ...
        (Xtr - muCV) ./ sigmaCV;

    XteStd = ...
        (Xte - muCV) ./ sigmaCV;


    %% ============================================================
    % DECISION TREE
    % ============================================================

    dtCV = fitctree( ...
        Xtr, ...
        Ytr, ...
        'MaxNumSplits',12, ...
        'MinLeafSize',5);

    predDT = ...
        predict(dtCV, Xte);

    cvAccuracyDT(fold) = ...
        mean(predDT == Yte) * 100;


    %% ============================================================
    % KNN
    % ============================================================

    knnCV = fitcknn( ...
        XtrStd, ...
        Ytr, ...
        'NumNeighbors',5, ...
        'Distance','euclidean', ...
        'Standardize',false);

    predKNN = ...
        predict(knnCV, XteStd);

    cvAccuracyKNN(fold) = ...
        mean(predKNN == Yte) * 100;


    %% ============================================================
    % SVM
    % ============================================================

    svmCVTemplate = templateSVM( ...
        'KernelFunction','linear', ...
        'Standardize',false);

    svmCV = fitcecoc( ...
        XtrStd, ...
        Ytr, ...
        'Learners',svmCVTemplate);

    predSVM = ...
        predict(svmCV, XteStd);

    cvAccuracySVM(fold) = ...
        mean(predSVM == Yte) * 100;


    %% ============================================================
    % RANDOM FOREST
    % ============================================================

    rfCV = fitcensemble( ...
        Xtr, ...
        Ytr, ...
        'Method','Bag', ...
        'NumLearningCycles',50);

    predRF = ...
        predict(rfCV, Xte);

    cvAccuracyRF(fold) = ...
        mean(predRF == Yte) * 100;

end


%% ================================================================
% 32. CROSS VALIDATION SUMMARY
% ================================================================

cvMean = [ ...
    mean(cvAccuracyDT); ...
    mean(cvAccuracyKNN); ...
    mean(cvAccuracySVM); ...
    mean(cvAccuracyRF)];

cvStd = [ ...
    std(cvAccuracyDT); ...
    std(cvAccuracyKNN); ...
    std(cvAccuracySVM); ...
    std(cvAccuracyRF)];


fprintf('\n');

fprintf( ...
    '%-20s %15s %15s\n', ...
    'Algorithm', ...
    'Mean CV (%)', ...
    'Std Dev (%)');

fprintf('------------------------------------------------------------\n');

for i = 1:length(modelNames)

    fprintf( ...
        '%-20s %14.2f %14.2f\n', ...
        modelNames{i}, ...
        cvMean(i), ...
        cvStd(i));

end

fprintf('============================================================\n\n');


%% ================================================================
% 33. CROSS VALIDATION GRAPH
% ================================================================

figure('Name','5 Fold Cross Validation');

b = bar(cvMean);

ylabel('Mean Cross Validation Accuracy (%)');

xlabel('Machine Learning Algorithm');

title('5-Fold Cross Validation Accuracy');

set(gca, ...
    'XTick', 1:length(modelNames), ...
    'XTickLabel', modelNames);

ylim([0 105]);

grid on;


for i = 1:length(cvMean)

    value = cvMean(i);

    text( ...
        i, ...
        value / 2, ...
        sprintf('%.1f%%',value), ...
        'HorizontalAlignment','center', ...
        'VerticalAlignment','middle', ...
        'FontWeight','bold', ...
        'FontSize',9, ...
        'Color','white');

end


%% ================================================================
% 34. CROSS VALIDATION TABLE
% ================================================================

CrossValidationTable = table( ...
    modelNames, ...
    cvMean, ...
    cvStd, ...
    'VariableNames', ...
    {'Algorithm', ...
     'Mean_5Fold_Accuracy_Percent', ...
     'Std_Deviation_Percent'});


%% ================================================================
% 35. SAVE RESULTS TO EXCEL
% ================================================================

outputExcel = ...
    'SmartGridShield_ML_Algorithm_Comparison.xlsx';

fprintf('Saving results to Excel...\n');


writetable( ...
    ResultsTable, ...
    outputExcel, ...
    'Sheet', 'Performance');


writetable( ...
    ComplexityTable, ...
    outputExcel, ...
    'Sheet', 'Model_Complexity');


writetable( ...
    CrossValidationTable, ...
    outputExcel, ...
    'Sheet', '5Fold_CV');


FeatureImportanceTable = table( ...
    featureNames(:), ...
    importanceValues(:), ...
    'VariableNames', ...
    {'Feature','Importance'});


writetable( ...
    FeatureImportanceTable, ...
    outputExcel, ...
    'Sheet', 'Feature_Importance');


%% ================================================================
% 36. SAVE TRAINED MODELS
% ================================================================

outputMAT = ...
    'SmartGridShield_All_ML_Models.mat';

fprintf('Saving trained models...\n');

save( ...
    outputMAT, ...
    'treeModel', ...
    'knnModel', ...
    'svmModel', ...
    'rfModel', ...
    'featureNames', ...
    'mu', ...
    'sigma');


%% ================================================================
% 37. FINAL SUMMARY
% ================================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('                    FINAL SUMMARY\n');
fprintf('============================================================\n\n');


for i = 1:length(modelNames)

    fprintf( ...
        '%-20s\n', ...
        modelNames{i});

    fprintf( ...
        '   Training Accuracy : %.2f%%\n', ...
        trainingAccuracy(i));

    fprintf( ...
        '   Testing Accuracy  : %.2f%%\n', ...
        testingAccuracy(i));

    fprintf( ...
        '   Precision         : %.2f%%\n', ...
        precisionValues(i));

    fprintf( ...
        '   Recall            : %.2f%%\n', ...
        recallValues(i));

    fprintf( ...
        '   F1 Score          : %.2f%%\n', ...
        f1Values(i));

    fprintf( ...
        '   5-Fold CV         : %.2f%% +/- %.2f%%\n\n', ...
        cvMean(i), ...
        cvStd(i));

end


%% ================================================================
% 38. BEST MODEL
% ================================================================

[bestAccuracy, bestIndex] = ...
    max(testingAccuracy);


fprintf('============================================================\n');

fprintf( ...
    'Best Testing Accuracy Model : %s\n', ...
    modelNames{bestIndex});

fprintf( ...
    'Best Testing Accuracy       : %.2f%%\n', ...
    bestAccuracy);

fprintf('============================================================\n');


fprintf('\nResults saved:\n');

fprintf( ...
    '1. %s\n', ...
    outputExcel);

fprintf( ...
    '2. %s\n', ...
    outputMAT);


fprintf('\n');
fprintf('============================================================\n');
fprintf('                  PROGRAM COMPLETE\n');
fprintf('============================================================\n');


%% ================================================================
% 39. METRIC CALCULATION FUNCTION
% ================================================================

function [precisionMacro, recallMacro, f1Macro] = ...
    calculateMetrics(YTrue, YPred)


classes = categories(YTrue);

nClasses = numel(classes);


precision = zeros(nClasses,1);

recall = zeros(nClasses,1);

f1 = zeros(nClasses,1);


for k = 1:nClasses

    currentClass = classes{k};


    %% True Positive

    TP = sum( ...
        (YTrue == currentClass) & ...
        (YPred == currentClass));


    %% False Positive

    FP = sum( ...
        (YTrue ~= currentClass) & ...
        (YPred == currentClass));


    %% False Negative

    FN = sum( ...
        (YTrue == currentClass) & ...
        (YPred ~= currentClass));


    %% Precision

    if TP + FP == 0

        precision(k) = 0;

    else

        precision(k) = ...
            TP / (TP + FP);

    end


    %% Recall

    if TP + FN == 0

        recall(k) = 0;

    else

        recall(k) = ...
            TP / (TP + FN);

    end


    %% F1 Score

    if precision(k) + recall(k) == 0

        f1(k) = 0;

    else

        f1(k) = ...
            2 * precision(k) * recall(k) / ...
            (precision(k) + recall(k));

    end

end


%% Macro Average

precisionMacro = ...
    mean(precision) * 100;

recallMacro = ...
    mean(recall) * 100;

f1Macro = ...
    mean(f1) * 100;

end