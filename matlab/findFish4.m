
% Set the folder path
% folderPath = 'C:\Users\rober\Dropbox\WORK\Jeremy\concentricOMR\imgs';
folderPath = 'C:\Users\johnsonr\Dropbox\WORK\Jeremy\concentricOMR\imgs';

% Get a list of all JPG files in the folder
imageFiles = dir(fullfile(folderPath, '*.jpg'));  % change to *.png or *.tif as needed

% Preallocate cell array to hold the images
numImages = length(imageFiles);

% Extract numeric values from filenames
numericNames = zeros(length(imageFiles), 1);
for i = 1:length(imageFiles)
    % Remove extension
    [~, name, ~] = fileparts(imageFiles(i).name);
    % Convert to number (assumes the name is a number)
    numericNames(i) = str2double(name);
end

% Sort based on numeric values
[~, sortIdx] = sort(numericNames);
imageFiles = imageFiles(sortIdx);  % apply the sort

clear i name numericNames sortIdx

%% load 100 random images and calculate modes for background subtraction

rng('default');
r = randperm(numImages);
r = r(1:100);

full_img_sz = [4512, 4512];
ds_img_sz = [640, 640];

rand_imgs_full = zeros([full_img_sz, length(r)], 'uint8');
rand_imgs_ds = zeros([ds_img_sz, length(r)], 'uint8');

for i = 1:length(r)
    ri = r(i);
    imagePath = fullfile(imageFiles(ri).folder, imageFiles(ri).name);
    img = imread(imagePath);
    rand_imgs_full(:,:,i) = img(:,:,1);
    rand_imgs_ds(:,:,i) = imresize(img(:,:,1), ds_img_sz, 'bilinear');
    disp(i)
end

background_full = mode(rand_imgs_full, 3);
background_ds = mode(rand_imgs_ds, 3);

clear r ri img rand* i imagePath folderPath

%% find the fish in every frame of the downsampled images

ds_thresh = 55;
se1 = strel("disk", 1);
se2 = strel("disk", 2);
se4 = strel("disk", 4);
stats = {};
bbox_width = 0.0171875; % 11/640 (width of bbox relative to image width)
bbox_height = 0.0171875;

yolo = struct;
roi_sz = [320, 320];
roi_halfwidth = floor(roi_sz ./ 2);
roi_thresh = 115;

full_image_path = 'C:\Users\johnsonr\Dropbox\WORK\Jeremy\concentricOMR\yolo_data3\full\images';
full_label_path = 'C:\Users\johnsonr\Dropbox\WORK\Jeremy\concentricOMR\yolo_data3\full\labels';
roi_image_path = 'C:\Users\johnsonr\Dropbox\WORK\Jeremy\concentricOMR\yolo_data3\roi\images';
roi_label_path = 'C:\Users\johnsonr\Dropbox\WORK\Jeremy\concentricOMR\yolo_data3\roi\labels';



for i = 1:numImages
    imagePath = fullfile(imageFiles(i).folder, imageFiles(i).name);
    img_from_file = imread(imagePath);
    img_full = img_from_file(:,:,1);
    img_ds = imresize(img_full(:,:,1), ds_img_sz, 'bilinear');

    % find fish in downsampled image

    diff_ds = background_ds - img_ds;
    im_ds = diff_ds >= ds_thresh;
    im_ds = imerode(im_ds, se1);
    im_ds = imdilate(im_ds, se4);
    ds_stat = regionprops(im_ds, 'Area', 'Centroid', 'Orientation');

    thresh = ds_thresh;
    while(length(ds_stat) < 1)
        thresh = thresh -5;
        im_ds = diff_ds >= ds_thresh;
        im_ds = imerode(im_ds, se1);
        im_ds = imdilate(im_ds, se4);
        ds_stat = regionprops(im_ds, 'Area', 'Centroid', 'Orientation');
        disp('no fish found -- adjusting threshold')
    end

    while(length(ds_stat) > 1)
        im_ds = imdilate(im_ds, se1);
        ds_stat = regionprops(im_ds, 'Area', 'Centroid', 'Orientation');
        disp('too many connected components found -- imdilate used again');
    end

    % process roi in full image


    ds_centroid = ds_stat.Centroid ./ ds_img_sz;
    full_centroid_px = round(ds_centroid .* full_img_sz);
    x1 = full_centroid_px(1) - roi_halfwidth(2);
    x2 = x1 + roi_sz(2) - 1;
    y1 = full_centroid_px(2) - roi_halfwidth(1);
    y2 = y1 + roi_sz(1) - 1;

    roi = img_full(y1:y2, x1:x2);
    diff_roi = background_full(y1:y2, x1:x2) - roi;
    im_roi = diff_roi >= roi_thresh;
    im_roi = imerode(im_roi, se1);
    im_roi = imdilate(im_roi, se2);
    im_roi = imerode(im_roi, se1);

    L = bwlabel(im_roi);
    roi_stat = regionprops(L, 'Area');
    [S, I] = sort([roi_stat.Area], 'descend');

    while(length(S) < 3)
        im_roi = imerode(im_roi, se1);
        L = bwlabel(im_roi);
        roi_stat = regionprops(L, 'Area');
        [S, I] = sort([roi_stat.Area], 'descend');
    end

    % keep the 3 largest connected components
    im = zeros(size(im_roi), 'logical');
    im(L == I(1)) = 1;
    im(L == I(2)) = 1;
    im(L == I(3)) = 1;

    L3 = bwlabel(im);
    roi_stat = regionprops(im, 'Area', 'Centroid', 'Orientation');
    
    pts = [roi_stat(1).Centroid; roi_stat(2).Centroid; roi_stat(3).Centroid];

    [angles, side_lengths] = triangle_calculations(pts(1,:), pts(2,:), pts(3,:));
    
    % kp_idx(1) is the index for the swim bladder
    % kp_idx(2:3) are the indices for the eyes
    [~, kp_idx] = sort(angles, 'ascend');

    % find midpoint between eyes
    eyeMean = mean(pts(kp_idx(2:3), :), 1);
    pts_centered = pts - eyeMean;

    head_vec = eyeMean - pts(kp_idx(1),:);
    heading = cart2pol(head_vec(1), -head_vec(2));
    % trans_vec = (roi_sz ./ 2) - eyeMean;
    % im_ref = imtranslate(roi, trans_vec);
    % im_ref = imrotate(im_ref, -heading * (180/pi));

    % Create rotation matrix
    R = [cos(heading) -sin(heading); sin(heading) cos(heading)];
    % Rotate your point(s)
    rotpts = (R*pts_centered')';
    
    cc_idx = zeros(1,3);
    % set swim bladder index
    cc_idx(1) = kp_idx(1);
    
    eye1 = kp_idx(2);
    eye2 = kp_idx(3);
    if rotpts(eye1, 2) > 0
        cc_idx(2) = eye1;
        cc_idx(3) = eye2;
    else
        cc_idx(2) = eye2;
        cc_idx(3) = eye1;
    end

    bladder = roi_stat(cc_idx(1));
    eyeL    = roi_stat(cc_idx(2));
    eyeR    = roi_stat(cc_idx(3));

    ds_label = [0, ds_centroid, bbox_width, bbox_height];
    ds_label_strings = string(ds_label);
    ds_label_string = join(ds_label_strings, " ");

    bladder_label = [0, bladder.Centroid ./ roi_sz];
    bladder_strings = string(bladder_label);
    bladder_string = join(bladder_strings, " ");

    eyeL_label = [1, eyeL.Centroid ./ roi_sz];
    eyeL_strings = string(eyeL_label);
    eyeL_string = join(eyeL_strings, " ");

    eyeR_label = [2, eyeR.Centroid ./ roi_sz];
    eyeR_strings = string(eyeR_label);
    eyeR_string = join(eyeR_strings, " ");

    % write out files for full image and label
    igm2write = zeros([ds_img_sz, 3], 'uint8');
    img2write(:,:,1) = img_ds;
    img2write(:,:,2) = img_ds;
    img2write(:,:,3) = img_ds;
    img_ds_filename = full_image_path + "\" + num2str(i) + ".png"; 
    imwrite(img2write, img_ds_filename);

    label_filename = full_label_path + "\" + num2str(i) + ".txt";
    writelines(ds_label_string, label_filename);

    % write out files for roi image and label
    roi2write = zeros([roi_sz, 3], 'uint8');
    roi2write(:,:,1) = roi;
    roi2write(:,:,2) = roi;
    roi2write(:,:,3) = roi;

    roi_filename = roi_image_path + "\" + num2str(i) + ".png"; 
    imwrite(roi2write, roi_filename);

    roi_label_filename = roi_label_path + "\" + num2str(i) + ".txt";
    roi_label_string = join([bladder_string; eyeL_string; eyeR_string], " ");
    writelines(roi_label_string, roi_label_filename);
    if (mod(i,5) == 0)
        disp(i)
    end
end

%%

writePath = 'C:\Users\johnsonr\Dropbox\WORK\Jeremy\concentricOMR\videos\fishtrack_test1.avi';
v = VideoWriter(writePath,'Motion JPEG AVI');
v.Quality = 80;
open(v);

figure;
for i = 1:numImages
    imshow(yolo(i).roi, [0 255]);
    hold on;
    scatter(yolo(i).bladder.Centroid(1), yolo(i).bladder.Centroid(2), 'pentagram', 'MarkerEdgeColor', 'k',...
        'MarkerFaceColor', 'y');

    scatter(yolo(i).eyeL.Centroid(1), yolo(i).eyeL.Centroid(2), 'o', 'MarkerEdgeColor', 'k',...
        'MarkerFaceColor', 'blue');

    scatter(yolo(i).eyeR.Centroid(1), yolo(i).eyeR.Centroid(2), 'o', 'MarkerEdgeColor', 'k',...
        'MarkerFaceColor', 'red');

    frame = getframe(gcf);
    writeVideo(v,frame)
    hold off;
    disp(i);
end

close(v);

