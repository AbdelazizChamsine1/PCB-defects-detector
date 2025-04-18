%% 1. Set Paths and Load Annotations
dataDir = fullfile("C:\Users\USER\Desktop\USJ\sem 6\Matlab\matlab_mid");
imageDir = fullfile(dataDir, "PCB-DATASET-master", "images");
annoDir  = fullfile(dataDir, "PCB-DATASET-master", "Annotations");

imds = imageDatastore(imageDir, ...
    FileExtensions=".jpg", IncludeSubfolders=true);

fds = fileDatastore(annoDir, ...
    "ReadFcn", @readPCBDefectAnnotations, ...
    "FileExtensions", ".xml", IncludeSubfolders=true);

annotations = readall(fds);

filenames = strings(0);
allBoxes = cell(0);
allLabels = cell(0);

for i = 1:numel(annotations)
    if ~isempty(annotations{i}.filename) && ~isempty(annotations{i}.Boxes)
        filenames(end+1) = annotations{i}.filename;
        allBoxes{end+1} = annotations{i}.Boxes;
        allLabels{end+1} = annotations{i}.Labels;
    end
end

tbl = table(filenames', allBoxes', allLabels', 'VariableNames', {'filename', 'Boxes', 'Labels'});
blds = boxLabelDatastore(tbl(:, {'Boxes', 'Labels'}));
ds = combine(imageDatastore(tbl.filename), blds);

%% 2. Preview an Annotated Image
if height(tbl) > 0
    img = imread(tbl.filename{1});
    bbox = tbl.Boxes{1};
    label = tbl.Labels{1};
    annotated = insertObjectAnnotation(img, 'rectangle', bbox, label);
    imshow(annotated);
    title("Sample Annotated PCB Image");
end

%% 3. Split Dataset (Train / Val / Test)
allLabelList = vertcat(tbl.Labels{:});
classNames = categories(categorical(allLabelList));
for i = 1:height(tbl)
    tbl.Labels{i} = categorical(tbl.Labels{i}, classNames);
end
rng("default");
numFiles = height(tbl);
shuffledIdx = randperm(numFiles);

numTrain = floor(0.7 * numFiles);
numVal   = floor(0.15 * numFiles);

tblTrain = tbl(shuffledIdx(1:numTrain), :);
tblVal   = tbl(shuffledIdx(numTrain+1:numTrain+numVal), :);
tblTest  = tbl(shuffledIdx(numTrain+numVal+1:end), :);

imdsTrain = imageDatastore(tblTrain.filename);
imdsVal   = imageDatastore(tblVal.filename);
imdsTest  = imageDatastore(tblTest.filename);

bldsTrain = boxLabelDatastore(tblTrain(:, {'Boxes', 'Labels'}));
bldsVal   = boxLabelDatastore(tblVal(:, {'Boxes', 'Labels'}));
bldsTest  = boxLabelDatastore(tblTest(:, {'Boxes', 'Labels'}));

dsTrain = combine(imdsTrain, bldsTrain);
dsVal   = combine(imdsVal, bldsVal);
dsTest  = combine(imdsTest, bldsTest);

%% 4. Define YOLOX Detector Architecture
%classNames = categories(tbl.Labels{1});
inputSize = [640 640 3];  % You can use [416 416 3] for faster training

detectorIn = yoloxObjectDetector("tiny-coco", classNames, InputSize=inputSize);

disp(categories(tbl.Labels{1}));
disp(categories(tbl.Labels{20}));

disp("YOLOX Class Names:");
disp(classNames);

disp("Example Labels from Training Set:");
disp(tblTrain.Labels{1});


%% 5. Specify Training Options
options = trainingOptions("sgdm", ...
    InitialLearnRate=5e-4, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropFactor=0.99, ...
    LearnRateDropPeriod=1, ...
    MiniBatchSize=8, ...
    MaxEpochs=10, ...
    Shuffle="every-epoch", ...
    VerboseFrequency=10, ...
    ValidationFrequency=100, ...
    ValidationData=dsVal, ...
    ResetInputNormalization=false, ...
    OutputNetwork="best-validation-loss");

%% 6. Train the YOLOX Detector
doTraining = true;
if doTraining
    [detector, info] = trainYOLOXObjectDetector(dsTrain, detectorIn, options);
    save("trainedPCBDetector.mat", "detector");
else
    load("trainedPCBDetector.mat");
end

%% 7. Evaluate Model on Test Set
disp("Running detection on test set...");

% Detect defects in test images
detectionResults = detect(detector, dsTest, Threshold=0.01);

% Evaluate results: calculates precision, recall, AP, mAP
metrics = evaluateObjectDetection(detectionResults, dsTest);

% Display Average Precision for each class
AP = averagePrecision(metrics);
apTable = table(classNames, AP);
disp("Average Precision per Class:");
disp(apTable);

% Compute mean Average Precision (mAP)
mAP = mean(AP);
fprintf("Mean Average Precision (mAP): %.4f\n", mAP);

%% 8. Plot Precision-Recall Curve for a Class
targetClass = classNames{4};  % You can change this to any class
[precision, recall, ~] = precisionRecall(metrics, ClassName=targetClass);

figure;
plot(recall{:}, precision{:}, 'LineWidth', 2);
title(sprintf("Precision-Recall Curve - Class: %s", targetClass));
xlabel("Recall");
ylabel("Precision");
grid on;

%% 9. Visualize Predictions on a Test Image
img = imread(tblTest.filename{55});
[bboxes, scores, labels] = detect(detector, img, Threshold=0.3);  % Show only confident ones
annotatedImg = insertObjectAnnotation(img, 'rectangle', bboxes, cellstr(labels));
figure;
imshow(annotatedImg);
title("YOLOX Predicted Defects on Test Image");

%% 10. Compare YOLOX Predictions vs Ground Truth

% Choose a test image index
idx = 3;  % or use: idx = randi(height(tblTest));
imgPath = tblTest.filename{idx};
img = imread(imgPath);

% Ground truth data
trueBBoxes = tblTest.Boxes{idx};
trueLabels = tblTest.Labels{idx};

% YOLOX predicted data
[predBBoxes, scores, predLabels] = detect(detector, img, Threshold=0.3);

% Annotate ground truth (green boxes)
imgGT = insertObjectAnnotation(img, 'rectangle', trueBBoxes, ...
    cellstr(trueLabels), 'Color', 'green');

% Annotate predictions (red boxes)
imgPred = insertObjectAnnotation(img, 'rectangle', predBBoxes, ...
    cellstr(predLabels), 'Color', 'red');

% Overlay: Display side-by-side
figure;
subplot(1,2,1);
imshow(imgGT);
title("Ground Truth (Green)");

subplot(1,2,2);
imshow(imgPred);
title("YOLOX Predictions (Red)");
