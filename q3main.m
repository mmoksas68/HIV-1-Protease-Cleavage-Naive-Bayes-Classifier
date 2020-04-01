training = load('q2_train_set.txt');

%laplace smoothing 75 rows
%X = training(1:75, 1:160); y = training(1:75, 161);

X = training(:, 1:160); y = training(:, 161);

testData = load('q2_test_set.txt');
testX = testData(:, 1:160); testY = testData(:, 161);

% Question 3.1 
% -----------------------------------------
[N,N1,N0, Pi0, Pi1, T, T0, T1, theta0, theta1] = trainData(X,y);

[tp, tn, fp, fn, accuracy] = test(theta0, theta1, testX, testY, Pi0, Pi1);

fprintf("\nQuestion 3.1 results: \n\n");
fprintf("true positive: %d  true negative: %d\nfalse positive: %d  false negative: %d\n", tp, tn, fp, fn);
fprintf("accuracy: %f\n", accuracy);
fprintf("-----------------------------------------\n");

% Question 3.2 and Question 3.3
% -----------------------------------------
txt = fileread('q2_gag_sequence.txt');

xGag = preprocessGag(txt);
writematrix(xGag, 'one_hot_encoded_gag.txt');
[cleaveIndices, highest1, lowest0] = testGag(theta0, theta1, xGag, Pi0, Pi1);

mer8_1 = txt(highest1 : highest1+7);
mer8_0 = txt(lowest0 : lowest0+7);

fprintf("\nQuestion 3.2 results: \n \n");
tempSize = size(cleaveIndices);

for index = 1:tempSize(2)
    fprintf("Indices to be cleaved: %d %d\n", cleaveIndices(1,index), cleaveIndices(1,index)+1);
end

fprintf("-----------------------------------------\n");

fprintf("\nQuestion 3.3 results: \n \n");
fprintf("The 8-mer labeled as class 1 with highest probability: %s\n", mer8_1);
fprintf("The 8-mer labeled as class 0 with lowest probability: %s\n", mer8_0);

fprintf("-----------------------------------------\n");

% Question 3.4
% -----------------------------------------

%Laplace smoothing

laplace_accurasies = zeros (11,1);

%Traing dataset for the first 75 row 
laplace_accurasies_75 = zeros (11,1);
[N_75,N1_75,N0_75, Pi0_75, Pi1_75, T_75, T0_75, T1_75, theta0_75, theta1_75] = trainData(X(1:75,:),y(1:75,:));

indices = [0,1,2,3,4,5,6,7,8,9,10];
for index = 0 : 10
    % Laplace smoothing 
    thetaLaplace0 = (T0' + index) / (N0 + 2*index);
    thetaLaplace1 = (T1' + index) / (N1 + 2*index);
        
    [~, ~, ~, ~, laplace_accuracy] = test(thetaLaplace0, thetaLaplace1, testX, testY, Pi0, Pi1);
    
    laplace_accurasies(index+1, 1) = laplace_accuracy;
    
    % Laplace smoothing for the first 75 rows
    thetaLaplace0_75 = (T0_75' + index) / (N0_75 + 2*index);
    thetaLaplace1_75 = (T1_75' + index) / (N1_75 + 2*index);
    
    [ltp_75, ltn_75, lfp_75, lfn_75, laplace_accuracy_75] = test(thetaLaplace0_75, thetaLaplace1_75, testX, testY, Pi0_75, Pi1_75);
    
    laplace_accurasies_75(index+1, 1) = laplace_accuracy_75;
end


figure, plot(indices, laplace_accurasies, indices, laplace_accurasies_75), xlabel("a values"), ylabel("accuracy"), legend('normal', 'first 75 rows'), title("Laplace Smoothing Accuracies"),
grid on, set(gca, 'FontSize', 15),
set(gcf, 'Position', [1400 100 1200 900])
clear *75 thetaLaplace* 


% Question 3.5
% -----------------------------------------
fprintf("\nQuestion 3.5 results: \n \n");

table = zeros(160,2,2);

for i = 1:160
    table(i,1,1) = (N0 - T0(1,i)) / N;
    table(i,1,2) = T0(1,i) / N;
    table(i,2,1) = (N1 - T1(1,i)) / N;
    table(i,2,2) = T1(1,i) / N;
end

[information] = mutual(table);

information_accurasies = zeros (160,1);
indices = [1, 2:160];
tempX = [];
tempTestX = [];
for i = 1:160
    tempX = [tempX  X(:,information(i,2))];
    tempTestX = [tempTestX testX(:, information(i,2))];
    [N,N1,N0, Pi0, Pi1, T, T0, T1, theta0, theta1] = trainData( tempX(:,1:i), y );
    [tp, tn, fp, fn, accuracy] = test(theta0, theta1, tempTestX(:,1:i), testY, Pi0, Pi1);
    information_accurasies(i,1) = accuracy; 
    fprintf("step %f accuracy result: %f\n", i, accuracy);
end

figure, plot(indices, information_accurasies), xlabel("step k"), ylabel("accuracy"), title("Mutual Information Accurasies for 160 steps"),
grid on, set(gca, 'FontSize', 15),
set(gcf, 'Position', [1400 100 1200 900]);

clear temp* indices index i highest1 lowest0 mer* table 
fprintf("-----------------------------------------\n");

% Question 3.6
% -----------------------------------------
fprintf("\nQuestion 3.6 results: \n \n");
columnMean = mean(X,1);
centerX = X - columnMean;
[U,S,V] = svd(centerX);

P = U*S;

L = (S.^2)./(length(centerX)-1); 

pve = diag(L./(sum(diag(L)/100)));

figure, plot([1:160], pve', '.'), xlabel('PCs'), ylabel('Varience (%)'), title('Proportion of Variance Explained for each PC')
grid on, set(gca, 'FontSize', 15),
set(gcf, 'Position', [1400 100 1200 900]);

figure, hold on
for index = 1:length(centerX)
    pc1 = P(index, 1);
    pc2 = P(index, 2);
    pc3 = P(index, 3);

    if(y(index,1) == 1)
        plot3(pc1, pc2, pc3, 'bo', 'LineWidth', 3);
    else
        plot3(pc1, pc2, pc3, 'rx', 'LineWidth', 3);
    end
end

xlabel('PC1'), ylabel('PC2'), zlabel('PC3'), title('Principal Component Analysis')
view(85,25), grid on, set(gca, 'FontSize', 15)
set(gcf, 'Position', [1400 100 1200 900])

fprintf("Proportion of variance explained (PVE) for the obtained principal components is: %.3f%%'", pve(1,1) + pve(2,1) + pve(3,1));


% Functions
%-----------------

function [N,N1,N0, Pi0, Pi1, T, T0, T1, theta0, theta1] = trainData(X,y)
%TRAINDATA this function trains the given data according to the given label
    N = length(y);
    N1 = sum(y(:) == 1);
    N0 = N - N1;

    Pi1 = N1/N;
    Pi0 = N0/N;

    A = X'*y;
    T = sum(X);
    T0 = T-A';
    T1 = A'; 

    theta0 = T0' / N0;
    theta1 = T1' / N1;
end

function [tp, tn, fp, fn, accuracy] = test(theta0, theta1, testX, testY, Pi0, Pi1)
%TEST this function tests the given test data according to the trained parameters
%and finds the accuracy along with true positive true negative false positive false negative 
    fp = 0;
    fn = 0;
    tp = 0;
    tn = 0;

    for index = 1:length(testY)
       currentTest = testX(index, :);
       p0 = log(Pi0);
       p1 = log(Pi1);
       for j = 1:length(currentTest)
           if(currentTest(1, j) == 1)
               p0 = p0 + log(theta0(j,1));
               p1 = p1 + log(theta1(j,1));
           end
       end

       if( p0 > p1 && testY(index,1) == 0)
           tn = tn+1;
       end

       if( p0 > p1 && testY(index,1) == 1)
           fn = fn+1;
       end

       if( p1 > p0 && testY(index,1) == 1)
           tp = tp+1;
       end

       if( p1 > p0 && testY(index,1) == 0)
           fp = fp+1;
       end

    end

    accuracy = (tp + tn) / (tp+tn+fp+fn);
end

function [cleaveIndices, highest1, highest0] = testGag(theta0, theta1, xGag, Pi0, Pi1)
%TESTGAG this function tests the given gag sequence according to the trained parameters
%and finds the indices to be cleaved 
    tempIndeces = zeros(1, size(xGag,1));
    count = 0;
    max1 = -Inf;
    min0 = +Inf;
    highest1 = 0;
    highest0 = 0;
    for index = 1:size(xGag,1)
       currentTest = xGag(index, :);
       p0 = log(Pi0);
       p1 = log(Pi1);

       for j = 1:length(currentTest)
           if(currentTest(1, j) == 1)

               p0 = p0 + log(theta0(j,1));
               p1 = p1 + log(theta1(j,1));
           end
       end
       if( p1 > p0)
           count = count + 1;
           tempIndeces(1,count) = index+2;
           if p1 > max1
                max1 = p1;
                highest1 = index;
           end
       end
       if( p0 > p1)
           if p0 < min0
                min0 = p0;
                highest0 = index;
           end
       end
    end

    cleaveIndices = tempIndeces(1, 1:count);
end

function xGag = preprocessGag(txt)
%PREPROCESSGAG this function modifies the given gag sequence to one hot encoded version 
    charToInt = containers.Map({'g','p','a','v', 'l','i','m','c','f','y','w','h','k','r','q','n','e','d','s','t'},{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19});
    oneHotEnc = zeros(20,20);
    for j = 1 : 20
        currentEnc = zeros(1,20);
        for i = 1:20
            if(j == i)
                currentEnc(i) = 1;
            end
        end
        oneHotEnc(1:20,j) = currentEnc;
    end
    
    xGag = zeros(length(txt)-7, 160);
   
    for index = 1:length(txt)-7
        current8mer = zeros(1, 160);
        for i = 0:7
            current8mer(1, i*20+1: i*20+20) = oneHotEnc(charToInt(txt(index+i))+1, 1:20);
        end
        xGag(index, 1:160) = current8mer;
    end
end

function [information] = mutual(table)
%MUTUAL this function finds the information of each feature with labels 
%according to the given table and orders them. 

    tempInfo = zeros(1,length(table));
    for i = 1: length(table)
        for x = 1:2
            for y = 1:2
                pX = table(i,x,y) + table(i,x,3-y);            
                pY = table(i,x,y) + table(i,3-x,y);
                if(pX*pY*table(i,x,y) ~= 0 ) 
                    tempInfo(1,i) = tempInfo(1,i) + (table(i,x,y)*log((table(i,x,y))/(pX*pY)));
                else
                    tempInfo(1,i) = +Inf;
                end
            end

        end
    end
    tempInfo = [tempInfo ; 1:160];
    information = sortrows(tempInfo', 1, 'descend');
end