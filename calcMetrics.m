function [precision, recall, f1] = calcMetrics(y_true, y_pred)
    confMatrix = confusionmat(y_true, y_pred);
    tp = confMatrix(2,2);
    fp = confMatrix(1,2);
    fn = confMatrix(2,1);
    tn = confMatrix(1,1);

    precision = tp / (tp + fp);
    recall = tp / (tp + fn);
    f1 = 2 * (precision * recall) / (precision + recall);
end