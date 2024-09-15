% Load the dataset

data= readtable("data.csv");
labels = data(:, 2);
data = table2array(data(:, 3:end));
labels.Var2(strcmp(labels.Var2, 'B')) = {1};
labels.Var2(strcmp(labels.Var2, 'M')) = {2};
labels = cell2mat(table2array(labels));
data = [data labels];

X = data(:, 1:end-1);
y = data(:, end);

% Get feature names
featureNames = data1.Properties.VariableNames(3:end-1); % Adjust based on actual data structure

% Initialize variables to store recall
qdaRecall = zeros(5, 1);
adaBoostRecall = zeros(5, 1);

% Perform 5-fold cross-validation
cv = cvpartition(y, 'KFold', 5);

for i = 1:cv.NumTestSets
    trainIdx = cv.training(i);
    testIdx = cv.test(i);
    
    X_train = X(trainIdx, :);
    y_train = y(trainIdx);
    X_test = X(testIdx, :);
    y_test = y(testIdx);
    
    % Quadratic Discriminant Analysis
    qdaModel = fitcdiscr(X_train, y_train, 'DiscrimType', 'quadratic');
    qdaPredictions = predict(qdaModel, X_test);
    qdaRecall(i) = recallMetric(y_test, qdaPredictions);
    
    % AdaBoost
    adaBoostModel = fitcensemble(X_train, y_train, 'Method', 'AdaBoostM1');
    adaBoostPredictions = predict(adaBoostModel, X_test);
    adaBoostRecall(i) = recallMetric(y_test, adaBoostPredictions);
end

% Calculate permutation importance for QDA
qdaFeatureImportance = permutationImportance(X, y, @fitcdiscr, 'quadratic', mean(qdaRecall));

% Calculate permutation importance for AdaBoost
adaBoostFeatureImportance = permutationImportance(X, y, @fitcensemble, 'AdaBoostM1', mean(adaBoostRecall));

% Plot feature importance for QDA
figure;
bar(qdaFeatureImportance);
xticks(1:numel(adaBoostFeatureImportance));
xticklabels(featureNames);
xtickangle(45);  % Rotate x-axis labels if necessary for better readability
xlabel('Features');
ylabel('Importance');
grid on;

% Plot feature importance for AdaBoost

figure;
bar(adaBoostFeatureImportance);
xticks(1:numel(adaBoostFeatureImportance));
xticklabels(featureNames);
xtickangle(45);  % Rotate x-axis labels if necessary for better readability
xlabel('Features');
ylabel('Importance');
title('AdaBoost Feature Importance');
grid on;grid on;