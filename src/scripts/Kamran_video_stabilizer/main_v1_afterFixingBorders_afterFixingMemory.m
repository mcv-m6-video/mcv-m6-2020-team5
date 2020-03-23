% Houman Kamran - hooman_kamran@yahoo.com
% Semester Project - EE7700 - Spring 2012 - Camera Shake Removal
% Creation date: April 11, 2012
% Last update: April 19, 2012


%%
% reading inputs

clear all;
close all;
clc;

pathToFiles = input('Enter the path to the folder containg the files: ' , 's');
addpath(pathToFiles);
pathToData = input('Enter the path to the folder containing data files: ' , 's');
addpath(pathToData);
nameOfInput = input('Enter the name of the video file (only .avi files): ' , 's');
obj = VideoReader(nameOfInput);


%%
% % preprocessing
% 
% % reading the .avi file and adding it to the workspace
% intendedFrame = 27;
% num = 3;
% video = read(obj , [intendedFrame , intendedFrame+num]);
% 
% % extracting the properties of the video sequence
% originalHeight = obj.Height;
% originalWidth = obj.Width;
% numOfFrames = obj.NumberOfFrames;
% 
% % edit the frames - changing them to gray scale - changing them to double - changing the size
% factor = 200/originalHeight;
% doubleGrayVideo(:,:,1) = double(imresize((rgb2gray(video(:,:,:,1))) , factor));
% doubleGrayVideo(:,:,2) = double(imresize((rgb2gray(video(:,:,:,1))) , factor));
% [height , width , depth] = size(doubleGrayVideo);


%%
% preprocessing

% reading the .avi file and adding it to the workspace
video = read(obj);
intendedFrame = 1;

% extracting the properties of the video sequence
originalHeight = obj.Height;
originalWidth = obj.Width;
numOfFrames = obj.NumberOfFrames;
num = numOfFrames-1;

% edit the frames - changing them to gray scale - changing them to double - changing the size
factor = 200/originalHeight;
doubleGrayVideo(:,:,1) = double(imresize((rgb2gray(video(:,:,:,1))) , factor));
doubleGrayVideo(:,:,2) = double(imresize((rgb2gray(video(:,:,:,1))) , factor));
[height , width , depth] = size(doubleGrayVideo);


%%
% body

% Create waitbar.
p = waitbar(0,'processing ...');
set(p,'Name','removing camera shake');

x = zeros(num,1);
y = zeros(num,1);
xSmoothMotion = zeros(num,1);
ySmoothMotion = zeros(num,1);
xShakeMotion = zeros(num,1);
yShakeMotion = zeros(num,1);
multiple = 0.5;
finalConvex = zeros(round((1+(2*multiple))*(height)) , round((1+(2*multiple))*(width)) , num+1);
[finalHeight , finalWidth , FinalDepth] = size(finalConvex);
startPoint(:,1) = round((multiple)*(height)); 
startPoint(:,2) = round((multiple)*(width));
maxStartPointHeight = startPoint(:,1);
minStartPointHeight = startPoint(:,1);
maxStartPointWidth = startPoint(:,2);
minStartPointWidth = startPoint(:,2);
finalConvex(startPoint(:,1)+1:startPoint(:,1)+height , startPoint(:,2)+1:startPoint(:,2)+width , 1) = doubleGrayVideo(:,:,1);
countShitySituation1 = 0;
countShitySituation11 = 0;
countShitySituation2 = 0;
countShitySituation21 = 0;

for k = 1:num
    
    if (mod(k,20) == 0)
        waitbar(k/(2*num));
    end
    
    % reading frames to be processed
    doubleGrayVideo(:,:,1) = doubleGrayVideo(:,:,2);
    doubleGrayVideo(:,:,2) = double(imresize((rgb2gray(video(:,:,:,k+1))) , factor));
    
    % finding the motion vectors from (i)th to (i+1)th
    [u1(:,:,k),v1(:,:,k)] = LucasKanadeHierarchical(doubleGrayVideo(:,:,2), doubleGrayVideo(:,:,1), 5, 5, 3);
    
    l = 0;
    m1 = zeros(2,1);
    m2 = zeros(2,1);
    % making two matrices of correspondances
    for i = 1:height
        for j = 1:width
            if ((u1(i,j,k) ~= 0) || (v1(i,j,k) ~= 0))
                l = l + 1;
                m1(: , l) = [i , j];
                m2(: , l) = [i+v1(i,j,k) , j+u1(i,j,k)];
            end
        end
    end
    
    % Using RANSAC to find outliers
    [H , inliers] = ransacfithomography_v2(m1(:,:) , m2(:,:) , 0.01);

    if (((sum((sum(H == eye(3))))==9) && (inliers == 0)) || ((sum((sum(H == eye(3))))==9) && (inliers == 1)))
        if (inliers == 0)
            countShitySituation1 = countShitySituation1 + 1;
        else
            countShitySituation11 = countShitySituation11 + 1;
        end
        if (k==1)
            x(k)=0;
            y(k)=0;
        else
            x(k) = x(k-1);
            y(k) = y(k-1);
        end
    else
        [numberOfRows , numberOfInliers] = size(inliers);

        % Making two matrices for inliers in first and second images
        for i = 1:numberOfInliers
            inliers1(:,i) = m1(:,inliers(1,i)); 
            inliers2(:,i) = m2(:,inliers(1,i)); 
        end
    
        % calculating the average movement
        for i = 1:numberOfInliers
            x(k) = x(k) + (inliers2(2,i) - inliers1(2,i));
            y(k) = y(k) + (inliers2(1,i) - inliers1(1,i));
        end
        x(k) = x(k)/numberOfInliers;
        y(k) = y(k)/numberOfInliers;
    end
    
end

xSmoothMotion = smooth(x,25);
ySmoothMotion = smooth(y,25);

for k = 1:num
    
    if (mod(k,20) == 0)
        waitbar((k+num)/(2*num));
    end
    
    l = 0;
    m1 = zeros(2,1);
    m2 = zeros(2,1);
    % making two matrices of correspondances
    for i = 1:height
        for j = 1:width
            if (((u1(i,j,k)-xSmoothMotion(k)) ~= 0) || ((v1(i,j,k)-ySmoothMotion(k)) ~= 0))
                l = l + 1;
                m1(: , l) = [i , j];
                m2(: , l) = [i+(v1(i,j,k)-ySmoothMotion(k)) , j+(u1(i,j,k)-xSmoothMotion(k))];
            end
        end
    end
    
    % Using RANSAC to find outliers
    [H , inliers] = ransacfithomography_v2(m1(:,:) , m2(:,:) , 0.01);

    if (((sum((sum(H == eye(3))))==9) && (inliers == 0)) || ((sum((sum(H == eye(3))))==9) && (inliers == 1)))
        if (inliers == 0)
            countShitySituation2 = countShitySituation2 + 1;
        else
            countShitySituation21 = countShitySituation21 + 1;
        end
        xShakeMotion(k)=0;
        yShakeMotion(k)=0;
    else
        [numberOfRows , numberOfInliers] = size(inliers);

        % Making two matrices for inliers in first and second images
        for i = 1:numberOfInliers
            inliers1(:,i) = m1(:,inliers(1,i)); 
            inliers2(:,i) = m2(:,inliers(1,i)); 
        end
    
        % calculating the average movement
        for i = 1:numberOfInliers
            xShakeMotion(k) = xShakeMotion(k) + (inliers2(2,i) - inliers1(2,i));
            yShakeMotion(k) = yShakeMotion(k) + (inliers2(1,i) - inliers1(1,i));
        end
        xShakeMotion(k) = xShakeMotion(k)/numberOfInliers;
        yShakeMotion(k) = yShakeMotion(k)/numberOfInliers;
    end

    startPoint(:,1) = startPoint(:,1) - round(yShakeMotion(k));
    startPoint(:,2) = startPoint(:,2) - round(xShakeMotion(k));
    if (startPoint(:,1) > maxStartPointHeight)
        maxStartPointHeight = startPoint(:,1);
    else
        if (startPoint(:,1) < minStartPointHeight)
            minStartPointHeight = startPoint(:,1);
        end
    end
    if (startPoint(:,2) > maxStartPointWidth)
        maxStartPointWidth = startPoint(:,2);
    else
        if (startPoint(:,2) < minStartPointWidth)
            minStartPointWidth = startPoint(:,2);
        end
    end
    finalConvex(startPoint(:,1)+1:startPoint(:,1)+height , startPoint(:,2)+1:startPoint(:,2)+width , k+1) = double(imresize((rgb2gray(video(:,:,:,k+1))) , factor));
    
end

% close wiatbar
close(p);

%%
% for display purposes

% displaying the output
doubleGrayVideoResize = zeros(size(finalConvex));
for i = 1:num+1
    doubleGrayVideoResize(((round((multiple)*height))+1):((round((multiple)*height))+height) , ((round((multiple)*width))+1):((round((multiple)*width))+width) , i) = double(imresize((rgb2gray(video(:,:,:,i))) , factor));
end
finalResult1 = [doubleGrayVideoResize finalConvex];
implay(uint8(finalResult1));

padHeight = 10;
padWidth = 10;
finalResult2 = zeros(padHeight+((minStartPointHeight+height)-((maxStartPointHeight+1)-1))+padHeight , padWidth+((minStartPointWidth+width)-((maxStartPointWidth+1)-1))+padWidth+((minStartPointWidth+width)-((maxStartPointWidth+1)-1))+padWidth , num+1);
finalResult2(padHeight+1:padHeight+((minStartPointHeight+height)-((maxStartPointHeight+1)-1)) , padWidth+1:padWidth+((minStartPointWidth+width)-((maxStartPointWidth+1)-1)) , :) = doubleGrayVideoResize(maxStartPointHeight+1:minStartPointHeight+height , maxStartPointWidth+1:minStartPointWidth+width , :);
finalResult2(padHeight+1:padHeight+((minStartPointHeight+height)-((maxStartPointHeight+1)-1)) , padWidth+((minStartPointWidth+width)-((maxStartPointWidth+1)-1))+padWidth+1:padWidth+((minStartPointWidth+width)-((maxStartPointWidth+1)-1))+padWidth+((minStartPointWidth+width)-((maxStartPointWidth+1)-1)) , :) = finalConvex(maxStartPointHeight+1:minStartPointHeight+height , maxStartPointWidth+1:minStartPointWidth+width , :);
implay(uint8(finalResult2));

