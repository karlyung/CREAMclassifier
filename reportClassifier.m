% Settings ================================================================

% set static random number generator seed for consistent testing
rng('default');

% narrow 53 classifications down to 13 for better results
reclassify = true;

% directory of folder where the reports are
% folderDir = "Reports/"; % default location of all reports
folderDir = "Reports by Publisher/CSB/"; % substitute for CSB reports only, more accurate results

% get "recommendations" and "lessons learned" sections only (slightly better results) or the entire doc
getImpSection = true;

% limit classification data and results to 1 and 0 only for better results, no > 1
limitResultToOneAndZero = true;
% End of Settings =========================================================

% call dataCompiler class for classification data table
classTable = dataCompiler(folderDir,getImpSection);

% code for the reclassify setting
if reclassify == true
    % get newClassGroup, where each element is the sum previous elements in
    % classGroup
    classGroup = [4 3 5 2 8 3 2 1 3 2 2 6 2 6 4];
    newClassGroup = [1];
    currentColNum = 1;
    for i = 1:numel(classGroup)
        currentColNum = currentColNum + classGroup(i);
        newClassGroup = [newClassGroup currentColNum];
    end
    
    % sum classification data in classTable into reclassCell according to
    % its classes
    classMat = cell2mat(classTable(:,4:end));
    reclassCell = [];
    for i = 1:size(classMat,1)
        currentRow = [];
        for j = 2:numel(newClassGroup)
            currentTotal = sum(classMat(i,newClassGroup(j - 1) + 1:newClassGroup(j) - 1));
            currentRow = [currentRow currentTotal];
        end
        reclassCell = [reclassCell;currentRow];
    end
    
    % limit data to 1 if limitResultToOneAndZero setting is true
    if limitResultToOneAndZero == true
        reclassCell(reclassCell > 1) = 1;
    end
else
    % get data from classTable directly if reclassify setting is false
    reclassCell = classTable(:,4:end);
end

debug = []; % array for debug info
mdlArray = {}; % array for trained SVM models

if ~iscell(reclassCell)
    reclassCell = num2cell(reclassCell);
end
newClassTable = [classTable(:,1:3) reclassCell]; % final, new classTable

% train and test a SVM model for each class, where the magic is
for i = 4:size(newClassTable,2)
    i
    
    % partition total data into a training set (90%) and a testing set (10%)
    cvp = cvpartition(cell2mat(newClassTable(:,i)),'Holdout',0.1);
    dataTrain = newClassTable(cvp.training,:);
    dataTest = newClassTable(cvp.test,:);
    
    % get the 3rd column, which is the doc text data
    textDataTrain = [dataTrain{:,3}]';
    textDataTest = [dataTest{:,3}]';
    
    % convert train text data to bagOfWords objects for input
    documents = preprocess(textDataTrain);
    bag = bagOfWords(documents);
    bag = removeInfrequentWords(bag,2);
    [bag,idx] = removeEmptyDocuments(bag);
    XTrain = bag.Counts;

    % convert test text data based on train bagOfWords
    documentsTest = preprocess(textDataTest);
    XTest = encode(bag,documentsTest);
    
    % get the current classification data
    YTrain = [dataTrain{:,i}]';
    YTest = [dataTest{:,i}]';
    
    % remove YTrain elements where docs are empty
    YTrain(idx) = [];
    
    % run SVM fitting process
    mdl = fitcecoc(XTrain,YTrain,'Learners','linear');
    
    % run more complex SVM, longer time, possibly better results
    % mdl = fitcecoc(XTrain,YTrain,'Learners',templateLinear('Solver','lbfgs'));
    
    % add current model to model array
    mdlArray{end + 1,1} = mdl;
    
    % get classification predictions using current model and test input data
    YPred = predict(mdl,XTest);
    
    % calculate prediction accuracy
    acc = sum(YPred == YTest)/numel(YTest);
    
    % add accuracy, predicted output, actual correct data to debug info
    % array
    debug = [debug;acc YPred' -1 YTest'];
end

% calculate mean of all accuracies
mean(debug(:,1))

% code for predicting the classification of one file
% temp = strsplit(extractFileText("Reports/AF447 2009 Rio-Paris.pdf"),"\n");
% str = strjoin(temp(1,7168:7608));
% documentsNew = preprocess(str);
% XNew = encode(bag,documentsNew);
% 
% result = zeros(1,size(mdlArray,2));
% for i = 1:size(mdlArray,2)
%     labelsNew = predict(mdlArray{1,i},XNew);
%     result(i) = labelsNew;
% end