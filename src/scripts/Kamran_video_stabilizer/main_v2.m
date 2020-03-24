% Houman Kamran - hooman_kamran@yahoo.com
% Semester Project - EE7700 - Spring 2012 - Camera Shake Removal
% Creation date: April 11, 2012
% Last update: April 17, 2012


%%
% reading inputs

clear all;
close all;
clc;

pathToFiles = input('Enter the path to the folder containg the files: ' , 's');
addpath(pathToFiles);
pathToData = input('Enter the path to the folder containing data files: ' , 's');
addpath(pathToData);

intendedFrame = 49;
num = 10;


%%
% preprocessing

% reading the .avi file and adding it to the workspace
nameOfInput = input('Enter the name of the video file (only .avi files): ' , 's');
obj = VideoReader(nameOfInput);
video = read(obj , [intendedFrame , intendedFrame+num]);

% extracting the properties of the video sequence
originalHeight = obj.Height;
originalWidth = obj.Width;
numOfFrames = obj.NumberOfFrames;

% edit the frames - changing them to gray scale - changing them to double - changing the size
factor = 200/originalHeight;
for i = 1:1+num
    doubleGrayVideo(:,:,i) = double(imresize((rgb2gray(video(:,:,:,i))) , factor));
end
[height , width] = size(doubleGrayVideo(:,:,1));


%%
% body

l = 0;
x = zeros(num,1);
y = zeros(num,1);
finalConvex = zeros(round((1.4)*(height)) , round((1.4)*(width)) , num+1);
[heightFinal , widthFinal , depthFinal] = size(finalConvex);
startPoint(:,1) = round((0.2)*(height)); 
startPoint(:,2) = round((0.2)*(width));
finalConvex(startPoint(:,1)+1:startPoint(:,1)+height , startPoint(:,2)+1:startPoint(:,2)+width , 1) = doubleGrayVideo(:,:,1);

for k = 1:num
    
    % finding the motion vectors from (i)th to (i+1)th
    [u1(:,:,k),v1(:,:,k)] = LucasKanadeHierarchical(doubleGrayVideo(:,:,k+1), doubleGrayVideo(:,:,k), 5, 5, 3);
    
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
    [H , inliers] = ransacfithomography(m1 , m2 , 0.01);
    [numberOfRows , numberOfInliers] = size(inliers);

    % Calculating new points in final images in final video
    for i = 1:heightFinal
        for j = 1:widthFinal
            newI = i - round((0.2)*(height));
            newJ = j - round((0.2)*(width));
            newPosition = H*[newI ; newJ ; 1];
            newXPosition = round(newPosition(1,1)/newPosition(3,1));
            newYPosition = round(newPosition(2,1)/newPosition(3,1));
%             newXPosition = round(newPosition(1,1));
%             newYPosition = round(newPosition(2,1));
            if ((1 <= newXPosition) && (newXPosition <= height) && (1 <= newYPosition) && (newYPosition <= width))
                finalConvex(i,j,k+1) = doubleGrayVideo(newXPosition,newYPosition,k+1);
            end
        end
    end
    
end


%%
% for display purposes

% displaying the output
implay(uint8(doubleGrayVideo));

figure; imshow(uint8(finalConvex(:,:,1)));
for i = 1:num
%     h = fspecial('gaussian',[3 3],0.5);
%     u_disp = imfilter(u1(:,:,i),h); u_disp = imresize(u_disp,0.25,'bilinear');
%     v_disp = imfilter(v1(:,:,i),h); v_disp = imresize(v_disp,0.25,'bilinear');
%     figure;
%     quiver(flipud(u_disp),-flipud(v_disp)); axis equal;
%     figure;
%     quiver(flipud(u1(:,:,i)),-flipud(v1(:,:,i))); axis equal;
    
%    [doubleGrayVideoWarped] = ImageWarp(doubleGrayVideo(:,:,i+1), u1(:,:,i), v1(:,:,i));
%    figure; imagesc(abs(doubleGrayVideo(:,:,i+1) - doubleGrayVideo(:,:,i))); title('Residual before motion compensation');
%    figure; imagesc(abs(doubleGrayVideoWarped - doubleGrayVideo(:,:,i))); title('Residual after motion compensation');
%    figure; imshow(uint8(doubleGrayVideo(:,:,i+1))); title('im1warped');
%    figure; imshow(uint8(doubleGrayVideo(:,:,i))); title('im1warped');
%    figure; imshow(uint8(doubleGrayVideoWarped)); title('im1warped');
    figure; imshow(uint8(finalConvex(:,:,i+1)));
end
