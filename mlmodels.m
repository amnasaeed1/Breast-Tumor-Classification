data = readtable("heart.csv");
labels = data(:, 12);

data = table2array(data);
labels.Var2(strcmp(labels.Var2, 'B')) = {1};
labels.Var2(strcmp(labels.Var2, 'M')) = {2};
labels = cell2mat(table2array(labels));
data = [data labels];

X = data(:, 1:end-1);
y = data(:, end);

% Initialize variables to store accuracies
logisticAccuracies = zeros(5, 1);
knnAccuracies = zeros(5, 1);
svmAccuracies = zeros(5, 1);
naiveBayesAccuracies = zeros(5, 1);
decisionTreeAccuracies = zeros(5, 1);
randomForestAccuracies = zeros(5, 1);
ldaAccuracies = zeros(5, 1);
qdaAccuracies = zeros(5, 1);
adaBoostAccuracies = zeros(5, 1);
gbmAccuracies = zeros(5, 1);

% Initialize variables to store precision, recall, F1 score, and AUC
logisticPrecision = zeros(5, 1);
logisticRecall = zeros(5, 1);
logisticF1 = zeros(5, 1);

knnPrecision = zeros(5, 1);
knnRecall = zeros(5, 1);
knnF1 = zeros(5, 1);

svmPrecision = zeros(5, 1);
svmRecall = zeros(5, 1);
svmF1 = zeros(5, 1);

naiveBayesPrecision = zeros(5, 1);
naiveBayesRecall = zeros(5, 1);
naiveBayesF1 = zeros(5, 1);

decisionTreePrecision = zeros(5, 1);
decisionTreeRecall = zeros(5, 1);
decisionTreeF1 = zeros(5, 1);

randomForestPrecision = zeros(5, 1);
randomForestRecall = zeros(5, 1);
randomForestF1 = zeros(5, 1);

ldaPrecision = zeros(5, 1);
ldaRecall = zeros(5, 1);
ldaF1 = zeros(5, 1);

qdaPrecision = zeros(5, 1);
qdaRecall = zeros(5, 1);
qdaF1 = zeros(5, 1);

adaBoostPrecision = zeros(5, 1);
adaBoostRecall = zeros(5, 1);
adaBoostF1 = zeros(5, 1);

gbmPrecision = zeros(5, 1);
gbmRecall = zeros(5, 1);
gbmF1 = zeros(5, 1);

% Variables to store all predictions and true labels
logisticAllPredictions = [];
knnAllPredictions = [];
svmAllPredictions = [];
naiveBayesAllPredictions = [];
decisionTreeAllPredictions = [];
randomForestAllPredictions = [];
ldaAllPredictions = [];
qdaAllPredictions = [];
adaBoostAllPredictions = [];
gbmAllPredictions = [];
allTrueLabels = [];

% Perform 5-fold cross-validation
cv = cvpartition(y, 'KFold', 5);

for i = 1:cv.NumTestSets
    trainIdx = cv.training(i);
    testIdx = cv.test(i);
    
    X_train = X(trainIdx, :);
    y_train = y(trainIdx);
    X_test = X(testIdx, :);
    y_test = y(testIdx);
    allTrueLabels = [allTrueLabels; y_test];

    % Logistic Regression
    model = fitclinear(X_train, y_train, 'Learner', 'logistic');
    predictions = predict(model, X_test);
    logisticAccuracies(i) = sum(predictions == y_test) / length(y_test);
    logisticAllPredictions = [logisticAllPredictions; predictions];
    
    % Calculate precision, recall, and F1 score for Logistic Regression
    [precision, recall, f1] = calcMetrics(y_test, predictions);
    logisticPrecision(i) = precision;
    logisticRecall(i) = recall;
    logisticF1(i) = f1;

    % k-Nearest Neighbors
    model = fitcknn(X_train, y_train);
    predictions = predict(model, X_test);
    knnAccuracies(i) = sum(predictions == y_test) / length(y_test);
    knnAllPredictions = [knnAllPredictions; predictions];
    
    % Calculate precision, recall, and F1 score for k-NN
    [precision, recall, f1] = calcMetrics(y_test, predictions);
    knnPrecision(i) = precision;
    knnRecall(i) = recall;
    knnF1(i) = f1;

    % Support Vector Machine
    model = fitcsvm(X_train, y_train);
    predictions = predict(model, X_test);
    svmAccuracies(i) = sum(predictions == y_test) / length(y_test);
    svmAllPredictions = [svmAllPredictions; predictions];
    
    % Calculate precision, recall, and F1 score for SVM
    [precision, recall, f1] = calcMetrics(y_test, predictions);
    svmPrecision(i) = precision;
    svmRecall(i) = recall;
    svmF1(i) = f1;

    % Naive Bayes
    model = fitcnb(X_train, y_train);
    predictions = predict(model, X_test);
    naiveBayesAccuracies(i) = sum(predictions == y_test) / length(y_test);
    naiveBayesAllPredictions = [naiveBayesAllPredictions; predictions];
    
    % Calculate precision, recall, and F1 score for Naive Bayes
    [precision, recall, f1] = calcMetrics(y_test, predictions);
    naiveBayesPrecision(i) = precision;
    naiveBayesRecall(i) = recall;
    naiveBayesF1(i) = f1;

    % Decision Tree
    model = fitctree(X_train, y_train);
    predictions = predict(model, X_test);
    decisionTreeAccuracies(i) = sum(predictions == y_test) / length(y_test);
    decisionTreeAllPredictions = [decisionTreeAllPredictions; predictions];
    
    % Calculate precision, recall, and F1 score for Decision Tree
    [precision, recall, f1] = calcMetrics(y_test, predictions);
    decisionTreePrecision(i) = precision;
    decisionTreeRecall(i) = recall;
    decisionTreeF1(i) = f1;

    % Random Forest
    model = TreeBagger(50, X_train, y_train, 'OOBPrediction', 'on', 'Method', 'classification');
    predictions = predict(model, X_test);
    predictions = str2double(predictions);
    randomForestAccuracies(i) = sum(predictions == y_test) / length(y_test);
    randomForestAllPredictions = [randomForestAllPredictions; predictions];
    
    % Calculate precision, recall, and F1 score for Random Forest
    [precision, recall, f1] = calcMetrics(y_test, predictions);
    randomForestPrecision(i) = precision;
    randomForestRecall(i) = recall;
    randomForestF1(i) = f1;

    % Linear Discriminant Analysis
    model = fitcdiscr(X_train, y_train);
    predictions = predict(model, X_test);
    ldaAccuracies(i) = sum(predictions == y_test) / length(y_test);
    ldaAllPredictions = [ldaAllPredictions; predictions];
    
    % Calculate precision, recall, and F1 score for LDA
    [precision, recall, f1] = calcMetrics(y_test, predictions);
    ldaPrecision(i) = precision;
    ldaRecall(i) = recall;
    ldaF1(i) = f1;

    % Quadratic Discriminant Analysis
    model = fitcdiscr(X_train, y_train, 'DiscrimType', 'quadratic');
    predictions = predict(model, X_test);
    qdaAccuracies(i) = sum(predictions == y_test) / length(y_test);
    qdaAllPredictions = [qdaAllPredictions; predictions];
    
    % Calculate precision, recall, and F1 score for QDA
    [precision, recall, f1] = calcMetrics(y_test, predictions);
    qdaPrecision(i) = precision;
    qdaRecall(i) = recall;
    qdaF1(i) = f1;

    % AdaBoost
    model = fitcensemble(X_train, y_train, 'Method', 'AdaBoostM1');
    predictions = predict(model, X_test);
    adaBoostAccuracies(i) = sum(predictions == y_test) / length(y_test);
    adaBoostAllPredictions = [adaBoostAllPredictions; predictions];
    
    % Calculate precision, recall, and F1 score for AdaBoost
    [precision, recall, f1] = calcMetrics(y_test, predictions);
    adaBoostPrecision(i) = precision;
    adaBoostRecall(i) = recall;
    adaBoostF1(i) = f1;

    % Gradient Boosting Machines (GBM)
    model = fitcensemble(X_train, y_train, 'Method', 'Bag');
    predictions = predict(model, X_test);
    gbmAccuracies(i) = sum(predictions == y_test) / length(y_test);
    gbmAllPredictions = [gbmAllPredictions; predictions];
    
    % Calculate precision, recall, and F1 score for GBM
    [precision, recall, f1] = calcMetrics(y_test, predictions);
    gbmPrecision(i) = precision;
    gbmRecall(i) = recall;
    gbmF1(i) = f1;
end

% Calculate and display the average accuracy, precision, recall, and F1 score for each model
modelNames = {'Logistic Regression', 'k-Nearest Neighbors', 'Support Vector Machine', 'Naive Bayes', ...
              'Decision Tree', 'Random Forest', 'Linear Discriminant Analysis', 'Quadratic Discriminant Analysis', ...
              'AdaBoost', 'Gradient Boosting Machines'};

averageAccuracies = [mean(logisticAccuracies), mean(knnAccuracies), mean(svmAccuracies), mean(naiveBayesAccuracies), ...
                     mean(decisionTreeAccuracies), mean(randomForestAccuracies), mean(ldaAccuracies), mean(qdaAccuracies), ...
                     mean(adaBoostAccuracies), mean(gbmAccuracies)];

averagePrecisions = [mean(logisticPrecision), mean(knnPrecision), mean(svmPrecision), mean(naiveBayesPrecision), ...
                     mean(decisionTreePrecision), mean(randomForestPrecision), mean(ldaPrecision), mean(qdaPrecision), ...
                     mean(adaBoostPrecision), mean(gbmPrecision)];

averageRecalls = [mean(logisticRecall), mean(knnRecall), mean(svmRecall), mean(naiveBayesRecall), ...
                  mean(decisionTreeRecall), mean(randomForestRecall), mean(ldaRecall), mean(qdaRecall), ...
                  mean(adaBoostRecall), mean(gbmRecall)];

averageF1s = [mean(logisticF1), mean(knnF1), mean(svmF1), mean(naiveBayesF1), ...
              mean(decisionTreeF1), mean(randomForestF1), mean(ldaF1), mean(qdaF1), ...
              mean(adaBoostF1), mean(gbmF1)];

fprintf('\nModel Performance:\n');
for i = 1:length(modelNames)
    fprintf('%s:\n', modelNames{i});
    fprintf('  Accuracy: %.4f\n', averageAccuracies(i));
    fprintf('  Precision: %.4f\n', averagePrecisions(i));
    fprintf('  Recall: %.4f\n', averageRecalls(i));
    fprintf('  F1 Score: %.4f\n\n', averageF1s(i));
end

avgLogisticAccuracy = mean(logisticAccuracies);
avgKnnAccuracy = mean(knnAccuracies);
avgSvmAccuracy = mean(svmAccuracies);
avgNaiveBayesAccuracy = mean(naiveBayesAccuracies);
avgDecisionTreeAccuracy = mean(decisionTreeAccuracies);
avgRandomForestAccuracy = mean(randomForestAccuracies);
avgLdaAccuracy = mean(ldaAccuracies);
avgQdaAccuracy = mean(qdaAccuracies);
avgAdaBoostAccuracy = mean(adaBoostAccuracies);
avgGbmAccuracy = mean(gbmAccuracies);


% Display confusion matrices
classes = {'Benign', 'Malignant'};

figure;
cm = confusionchart(allTrueLabels, logisticAllPredictions, 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
cm.Title = sprintf('Logistic Regression Accuracy: %.2f%%', avgLogisticAccuracy * 100);

figure;
cm = confusionchart(allTrueLabels, knnAllPredictions, 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
cm.Title = sprintf('k-NN Accuracy: %.2f%%', avgKnnAccuracy * 100);


figure;
cm = confusionchart(allTrueLabels, svmAllPredictions, 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
cm.Title = sprintf('SVM Accuracy: %.2f%%', avgSvmAccuracy * 100);


figure;
cm = confusionchart(allTrueLabels, naiveBayesAllPredictions, 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
cm.Title = sprintf('Naive Bayes Accuracy: %.2f%%', avgNaiveBayesAccuracy * 100);


figure;
cm = confusionchart(allTrueLabels, decisionTreeAllPredictions, 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
cm.Title = sprintf('Decision Tree Accuracy: %.2f%%', avgDecisionTreeAccuracy * 100);


figure;
cm = confusionchart(allTrueLabels, randomForestAllPredictions, 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
cm.Title = sprintf('Random Forest Accuracy: %.2f%%', avgRandomForestAccuracy * 100);


figure;
cm = confusionchart(allTrueLabels, ldaAllPredictions, 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
cm.Title = sprintf('LDA Accuracy: %.2f%%', avgLdaAccuracy * 100);


figure;
cm = confusionchart(allTrueLabels, qdaAllPredictions, 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
cm.Title = sprintf('QDA Accuracy: %.2f%%', avgQdaAccuracy * 100);


figure;
cm = confusionchart(allTrueLabels, adaBoostAllPredictions, 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
cm.Title = sprintf('AdaBoost Accuracy: %.2f%%', avgAdaBoostAccuracy * 100);


figure;
cm = confusionchart(allTrueLabels, gbmAllPredictions, 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
cm.Title = sprintf('GBM Accuracy: %.2f%%', avgGbmAccuracy * 100);

averageAccuracies = [mean(logisticAccuracies), mean(knnAccuracies), mean(svmAccuracies), mean(naiveBayesAccuracies), ...
                     mean(decisionTreeAccuracies), mean(randomForestAccuracies), mean(ldaAccuracies), mean(qdaAccuracies), ...
                     mean(adaBoostAccuracies), mean(gbmAccuracies)];

averagePrecisions = [mean(logisticPrecision), mean(knnPrecision), mean(svmPrecision), mean(naiveBayesPrecision), ...
                     mean(decisionTreePrecision), mean(randomForestPrecision), mean(ldaPrecision), mean(qdaPrecision), ...
                     mean(adaBoostPrecision), mean(gbmPrecision)];

averageRecalls = [mean(logisticRecall), mean(knnRecall), mean(svmRecall), mean(naiveBayesRecall), ...
                  mean(decisionTreeRecall), mean(randomForestRecall), mean(ldaRecall), mean(qdaRecall), ...
                  mean(adaBoostRecall), mean(gbmRecall)];

averageF1s = [mean(logisticF1), mean(knnF1), mean(svmF1), mean(naiveBayesF1), ...
              mean(decisionTreeF1), mean(randomForestF1), mean(ldaF1), mean(qdaF1), ...
              mean(adaBoostF1), mean(gbmF1)];

% Create a table with all the models and their metrics
T = table(modelNames', averageAccuracies', averagePrecisions', averageRecalls', averageF1s', ...
          'VariableNames', {'Model', 'Accuracy', 'Precision', 'Recall', 'F1_Score'});