import json
from io import BytesIO
from PIL import Image
import os

import streamlit as st
import pandas as pd
import numpy as np


from kao import *

from pathlib import Path
Path("ml_result").mkdir(parents=True, exist_ok=True)
Path("image_data").mkdir(parents=True, exist_ok=True)
Path("train_label_data").mkdir(parents=True, exist_ok=True)
Path("val_label_data").mkdir(parents=True, exist_ok=True)

# SETTING PAGE CONFIG TO WIDE MODE AND ADDING A TITLE AND FAVICON
st.set_page_config(layout="wide", page_title="AI-Sugarcane Classification", page_icon=":boy:")


st.title('AI-APP: Sugarcane Classification and Regression Model')


st.info('Assessment of machine learning on sugarcane classification')
#st.warning('Updated: December, 2022')

from PIL import Image
image = Image.open('logo6.png')

st.image(image, use_column_width=True)


first_stage, second_stage, third_stage = st.columns(3)


first_stage.header("**FIRST STAGE:** BROWSE IMAGES")

second_stage.header("**SECOND STAGE:** TRAIN ML MODEL")

third_stage.header("**THIRD STAGE:** EXPORT SHP FILE")


sat_image = first_stage.file_uploader('Upload Sattellite Image',type=["tif","tiff","png", "jpg", "jpeg"])
train_label_image_multiple = first_stage.file_uploader('Upload Shape File Labeling (For Training Model)', type=['shp','shx','xml','lock','sbx','sbn','prj','dbf','cpg'],accept_multiple_files=True)
val_label_image_multiple = first_stage.file_uploader('Upload Shape File Labeling (For Validation Model)',type=['shp','shx','xml','lock','sbx','sbn','prj','dbf','cpg'],accept_multiple_files=True)


#st.warning('Updated: December 2022')


# the remote sensing image you want to classify
if sat_image:
	#first_stage.write("Filename: ", sat_image.name)
	img_RS = sat_image.name   #  BROWSE YOUR IMAGE

	File = sat_image

	first_stage.markdown("**The file is sucessfully Uploaded.**")

	# Save uploaded file to 'F:/tmp' folder.

	save_folder = 'image_data/'
	save_path = Path(save_folder, File.name)

	with open(save_path, mode='wb') as w:
	#with open('input_image.tif', mode='wb') as w:
		w.write(File.getvalue())

	if save_path.exists():
		first_stage.success(f'File {File.name} is successfully saved!')

	img_raw = File.name.split('.')[0]+'.tif'

	img_RS = 'image_data/'+img_raw  #  BROWSE YOUR IMAGE


	#save_uploadedfile(sat_image)

if train_label_image_multiple:
	for train_label_image in train_label_image_multiple:
		#first_stage.write("Filename: ", train_label_image.name)
		training = train_label_image.name #  BROWSE YOUR IMAGE LABEL
		#save_uploadedfile(training)

		File = train_label_image

		first_stage.markdown("**The file is sucessfully Uploaded.**")

		# Save uploaded file to 'F:/tmp' folder.
		
		save_folder = 'train_label_data/'
		save_path = Path(save_folder, File.name)

		with open(save_path, mode='wb') as w:
		#with open('train.shp', mode='wb') as w:
			w.write(File.getvalue())

		if save_path.exists():
			first_stage.success(f'File {File.name} is successfully saved!')

	train_shape = File.name.split('.')[0]+'.shp'

	first_stage.success(f'~~ {train_shape} is final name saved!')

	print('FINAL TRAIN SHAPE:', train_shape)

	training = 'train_label_data/'+train_shape #  BROWSE YOUR IMAGE LABEL

		#save_uploadedfile(sat_image)

if val_label_image_multiple:
	for val_label_image in val_label_image_multiple:

		#first_stage.write("Filename: ", val_label_image.name)
		#save_uploadedfile(val_label_image)

		File = val_label_image

		first_stage.markdown("**The file is sucessfully Uploaded.**")

		# Save uploaded file to 'F:/tmp' folder.
		
		save_folder = 'val_label_data'
		save_path = Path(save_folder, File.name)

		with open(save_path, mode='wb') as w:
		#with open('val.shp', mode='wb') as w:
			w.write(File.getvalue())

		if save_path.exists():
			first_stage.success(f'File {File.name} is successfully saved!')


	val_shape = File.name.split('.')[0]+'.shp'

	validation = 'val_label_data/'+val_shape  #  BROWSE YOUR IMAGE LABEL


	first_stage.success(f'~~ {val_shape} is final name saved!')

	print('FINAL VAL SHAPE:', val_shape)


	#save_uploadedfile(sat_image)


# print('img_RS:',img_RS)
# print('training:',training)
# print('validation:',validation)

# the remote sensing image you want to classify
# img_RS = r'image_data/input_image.tif'   #  BROWSE YOUR IMAGE


# # training and validation as shape files
# training = train_shape #  BROWSE YOUR IMAGE LABEL
# validation = val_shape  #  BROWSE YOUR IMAGE LABEL


# st.sidebar.header("Customizing the model")
# ml_model_call = st.sidebar.selectbox("Select machine leanring model.",('Random Forest','Nueral Network','Gradient Boosting'))


ml_model_call = 'Random Forest'

SELECT = ml_model_call #'RF' # 'NN', 'GBT'

if SELECT == 'Random Forest':
    model = RandomForestClassifier(oob_score=True, verbose=True)
elif SELECT == 'Nueral Network':
    model = MLPClassifier(max_iter=1000, alpha=0.0001)
elif SELECT == 'Gradient Boosting':
    model = GradientBoostingClassifier(random_state=10)
    
print('YOUR MODE IS:', model)


if sat_image and train_label_image_multiple and val_label_image_multiple:
	
	print('OK LET STARTED')


	# define a number of trees that should be used (default = 500)
	est = 200

	# how many cores should be used?
	# -1 -> all available cores
	n_cores = -1
	#n_cores = 4


	# what is the attributes name of your classes in the shape file (field name of the classes)?
	attribute = 'LU_num'

	# directory, where the classification image should be saved:
	classification_image = r'ml_result/ml_class.tif'
	# directory, where the all meta results should be saved:
	results_txt = r'ml_result/ml_class.txt'

	# laod training data and show all shape attributes

	#model_dataset = gdal.Open(model_raster_fname)

	print()
	print('training path:', training);print()

	shape_dataset = ogr.Open(training)

	print()
	print('shape_dataset:', shape_dataset);print()

	shape_layer = shape_dataset.GetLayer()


	# extract the names of all attributes (fieldnames) in the shape file
	attributes = []
	ldefn = shape_layer.GetLayerDefn()
	for n in range(ldefn.GetFieldCount()):
	    fdefn = ldefn.GetFieldDefn(n)
	    attributes.append(fdefn.name)
	    
	# print the attributes
	print('Available attributes in the shape file are: {}'.format(attributes))

	# prepare results text file:

	print('Random Forest Classification', file=open(results_txt, "a"))
	print('Processing: {}'.format(datetime.datetime.now()), file=open(results_txt, "a"))
	print('-------------------------------------------------', file=open(results_txt, "a"))
	print('PATHS:', file=open(results_txt, "a"))
	print('Image: {}'.format(img_RS), file=open(results_txt, "a"))
	print('Training shape: {}'.format(training) , file=open(results_txt, "a"))
	print('Vaildation shape: {}'.format(validation) , file=open(results_txt, "a"))
	print('      choosen attribute: {}'.format(attribute) , file=open(results_txt, "a"))
	print('Classification image: {}'.format(classification_image) , file=open(results_txt, "a"))
	print('Report text file: {}'.format(results_txt) , file=open(results_txt, "a"))
	print('-------------------------------------------------', file=open(results_txt, "a"))

	# load image data

	img_ds = gdal.Open(img_RS, gdal.GA_ReadOnly)

	img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount),
	               gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
	for b in range(img.shape[2]):
	    img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()

	row = img_ds.RasterYSize
	col = img_ds.RasterXSize
	band_number = img_ds.RasterCount

	print('Image extent: {} x {} (row x col)'.format(row, col))
	print('Number of Bands: {}'.format(band_number))


	print('Image extent: {} x {} (row x col)'.format(row, col), file=open(results_txt, "a"))
	print('Number of Bands: {}'.format(band_number), file=open(results_txt, "a"))
	print('---------------------------------------', file=open(results_txt, "a"))
	print('TRAINING', file=open(results_txt, "a"))
	print('Number of Trees: {}'.format(est), file=open(results_txt, "a"))

	# laod training data from shape file

	#model_dataset = gdal.Open(model_raster_fname)
	shape_dataset = ogr.Open(training)
	shape_layer = shape_dataset.GetLayer()

	mem_drv = gdal.GetDriverByName('MEM')
	mem_raster = mem_drv.Create('',img_ds.RasterXSize,img_ds.RasterYSize,1,gdal.GDT_UInt16)
	mem_raster.SetProjection(img_ds.GetProjection())
	mem_raster.SetGeoTransform(img_ds.GetGeoTransform())
	mem_band = mem_raster.GetRasterBand(1)
	mem_band.Fill(0)
	mem_band.SetNoDataValue(0)

	att_ = 'ATTRIBUTE='+attribute
	# http://gdal.org/gdal__alg_8h.html#adfe5e5d287d6c184aab03acbfa567cb1
	# http://gis.stackexchange.com/questions/31568/gdal-rasterizelayer-doesnt-burn-all-polygons-to-raster
	err = gdal.RasterizeLayer(mem_raster, [1], shape_layer, None, None, [1],  [att_,"ALL_TOUCHED=TRUE"])
	assert err == gdal.CE_None

	roi = mem_raster.ReadAsArray()

	# Display images
	plt.subplot(121)
	plt.imshow(img[:, :, 0], cmap=plt.cm.Greys_r)
	plt.title('RS image - first band')
	#plt.imshow(img[:, :, 0:4], cmap='viridis')
	#plt.title('RS image - composite band')

	plt.subplot(122)
	plt.imshow(roi, cmap=plt.cm.Spectral)
	plt.title('Training Image')

	plt.show()

	# Number of training pixels:
	n_samples = (roi > 0).sum()
	print('{n} training samples'.format(n=n_samples))
	print('{n} training samples'.format(n=n_samples), file=open(results_txt, "a"))

	# What are our classification labels?
	labels = np.unique(roi[roi > 0])
	print('training data include {n} classes: {classes}'.format(n=labels.size, classes=labels))
	print('training data include {n} classes: {classes}'.format(n=labels.size, classes=labels), file=open(results_txt, "a"))

	# Subset the image dataset with the training image = X
	# Mask the classes on the training dataset = y
	# These will have n_samples rows
	X = img[roi > 0, :]
	y = roi[roi > 0]

	# Split training testing
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

	print('Our X matrix is sized: {sz}'.format(sz=X.shape))
	print('Our y array is sized: {sz}'.format(sz=y.shape))
	print('Our X_train matrix is sized: {sz}'.format(sz=X_train.shape))
	print('Our y_train array is sized: {sz}'.format(sz=y_train.shape))

	# Test model
	X_train = np.nan_to_num(X_train)
	#rf = GridSearchCV(rfc, parameter_space, n_jobs=n_cores, cv=5, verbose=10).fit(X_train, y_train)

	rf = model.fit(X_train, y_train)

	y_pred = rf.predict(X_test)
	print("Accuracy:", metrics.accuracy_score(y_test, y_pred))





	# Best paramete set
	#print('Best parameters found:\n', rf.best_params_)

	# All results
	# means = rf.cv_results_['mean_test_score']
	# stds = rf.cv_results_['std_test_score']
	# for mean, std, params in zip(means, stds, rf.cv_results_['params']):
	#     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

	# predict result
	print('Alternative displaying result')
	y_true, y_pred = y_test , rf.predict(X_test)

	from sklearn.metrics import classification_report
	print('Results on the test set:')
	print(classification_report(y_true, y_pred))


	#second_stage.markdown(classification_report(y_true, y_pred))

	cm = confusion_matrix(y_test,rf.predict(X_test))

	second_stage.warning('Performance Evaluation of Traning Model')


	second_stage.text('Model Report:\n ' + classification_report(y_true, y_pred))


	fig, ax = plt.subplots()
	sns.heatmap(cm, annot=True, cmap='Oranges', fmt='g', cbar=False)
	second_stage.write(fig)

	#second_stage.text('Confusion matrix: ', cm)
	# plt.figure(figsize=(10,7))
	# sn.heatmap(cm, annot=True, fmt='g')
	# plt.xlabel('classes - predicted')
	# plt.ylabel('classes - truth')
	# plt.show()

	# 2. Using pickle
	filename = r'ml_result/final_ml_model.pickle'
	pickle.dump(rf, open(filename, 'wb'))

	# inference load the model from disk
	rf = pickle.load(open(filename, 'rb'))
	result = rf.score(X_test, y_test)

	print(result)

	text_acc = 'Accuracy: '+str(metrics.accuracy_score(y_test, y_pred))
	text_pr = 'Precision: '+str(metrics.precision_score(y_test, y_pred,average='weighted'))
	text_re = 'Recall: '+str(metrics.recall_score(y_test, y_pred,average='weighted'))
	text_f1 = 'F1 Score: '+str(metrics.f1_score(y_test, y_pred,average='weighted'))

	second_stage.info(text_acc)
	second_stage.info(text_pr)
	second_stage.info(text_re)
	second_stage.info(text_f1)



	second_stage.warning('Regression Model Report')

	dat = r'regression/yield_data_edit.csv'
	df_reg = pd.read_csv(dat, header=0)
	df_reg.describe()

	second_stage.dataframe(df_reg.describe()) 

	df_reg = df_reg[(df_reg.tons_area<14.5) & (df_reg.tons_area>0) 
	& (df_reg.band1_ndvi>=0.7) & (df_reg.band2_gndvi<0.82) & (df_reg.band2_gndvi>0.65)
	 & (df_reg.band3_evi>0.4) & (df_reg.band4_savi>0.45) & (df_reg.band4_savi<0.72)]


	dfCor2 = df_reg.iloc[:,1:6]
	dfCor2.head()

	sns.pairplot(dfCor2)

	fig, ax = plt.subplots()
	sns.heatmap(dfCor2.corr(), ax=ax)
	second_stage.write(fig)

	# Variable definition
	ndvi = df_reg["band1_ndvi"].values
	gndvi = df_reg["band2_gndvi"].values
	evi = df_reg["band3_evi"].values
	savi = df_reg["band4_savi"].values
	FID = df_reg["FID_"].values

	X2 = np.array([ndvi, gndvi, evi, savi]).reshape(-1, 4)
	# X = np.array([ndvi, gndvi, savi]).reshape(-1, 3)
	y2 = df_reg["tons_area"].values

	X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.3, random_state=101)

	# Set cross validation
	cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)

	reg2 = LinearRegression().fit(X_train2, y_train2)
	#print('Regression cross validation score: ', np.mean(cross_val_score(reg, X_train, y_train, cv=cv)))
	#print('Regression score: ', reg.score(X_train, y_train))
	print('Regression coefficient: ', reg2.coef_)
	print('Intercept: ', reg2.intercept_)

	#print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
	y_pred2 = reg2.predict(X_test2)
	#prediction = np.concatenate((y_test.reshape(743,1), y_pred.reshape(743,1)), axis=1)
	print('r-squared: ', metrics.r2_score(y_test2, y_pred2))
	print('RMS Error:', np.sqrt(metrics.mean_squared_error(y_test2, y_pred2)))
	print('Mean Absolute Error:', metrics.mean_absolute_error(y_test2, y_pred2))
	print('Mean Squared Error:', metrics.mean_squared_error(y_test2, y_pred2))

	text_rsq = 'R-Squared: '+str(metrics.r2_score(y_test2, y_pred2))
	text_rms = 'RMS Error: '+str(np.sqrt(metrics.mean_squared_error(y_test2, y_pred2)))
	text_mae = 'Mean Absolute Error: '+str(metrics.mean_absolute_error(y_test2, y_pred2))
	text_mse = 'Mean Squared Error: '+str(metrics.mean_squared_error(y_test2, y_pred2))

	second_stage.info(text_rsq)
	second_stage.info(text_rms)
	second_stage.info(text_mae)
	second_stage.info(text_mse)





	third_stage.warning('Waiting for this process. This may take a while ...')


	########################################
	# ------------------------------------ #
	# ------------------------------------ #
	########################################


	########################################
	# ------------------------------------ #
	# ------------------------------------ #
	########################################


	import time
	start_time = time.time()

	time.sleep(30)

	# # Predicting the rest of the image

	# # Take our full image and reshape into long 2d array (nrow * ncol, nband) for classification
	# new_shape = (img.shape[0] * img.shape[1], img.shape[2])
	# img_as_array = img[:, :, :np.int(img.shape[2])].reshape(new_shape)
	# #img_as_array = img[:, :, :int(img.shape[2])].reshape(new_shape)

	# print('Reshaped from {o} to {n}'.format(o=img.shape, n=img_as_array.shape))

	# img_as_array = np.nan_to_num(img_as_array)

	# # Now predict for each pixel
	# # first prediction will be tried on the entire image
	# # if not enough RAM, the dataset will be sliced
	# try:
	#     class_prediction = rf.predict(img_as_array)
	# except MemoryError:
	#     #slices = int(round(len(img_as_array)/2))
	#     slices = int(round(len(img_as_array)/200))

	#     test = True
	    
	#     while test == True:
	#         try:
	#             class_preds = list()
	            
	#             temp = rf.predict(img_as_array[0:slices+1,:])
	#             class_preds.append(temp)
	            
	#             for i in range(slices,len(img_as_array),slices):
	#                 print('{} %, derzeit: {}'.format((i*100)/(len(img_as_array)), i))
	#                 temp = rf.predict(img_as_array[i+1:i+(slices+1),:])                
	#                 class_preds.append(temp)
	            
	#         except MemoryError as error:
	#             #slices = slices/2
	#             slices = slices/2
	#             print('Not enought RAM, new slices = {}'.format(slices))
	            
	#         else:
	#             test = False
	# else:
	#     print('Class prediction was successful without slicing!')
	    
	# # concatenate all slices and re-shape it to the original extend
	# try:
	#     class_prediction = np.concatenate(class_preds,axis = 0)
	# except NameError:
	#     print('No slicing was necessary!')
	    
	# class_prediction = class_prediction.reshape(img[:, :, 0].shape)
	# print('Reshaped back to {}'.format(class_prediction.shape))

	# # generate mask image from red band
	# mask = np.copy(img[:,:,0])
	# mask[mask > 0.0] = 1.0 # all actual pixels have a value of 1.0

	# # plot mask

	# plt.imshow(mask)

	# # mask classification an plot

	# class_prediction.astype(np.float16)
	# class_prediction_ = class_prediction*mask

	# plt.subplot(121)
	# plt.imshow(class_prediction, cmap=plt.cm.Spectral)
	# plt.title('classification unmasked')

	# plt.subplot(122)
	# plt.imshow(class_prediction_, cmap=plt.cm.Spectral)
	# plt.title('classification masked')

	# plt.show()

	# cols = img.shape[1]
	# rows = img.shape[0]

	# class_prediction_.astype(np.float16)

	# driver = gdal.GetDriverByName("gtiff")
	# outdata = driver.Create(classification_image, cols, rows, 1, gdal.GDT_UInt16)
	# outdata.SetGeoTransform(img_ds.GetGeoTransform())##sets same geotransform as input
	# outdata.SetProjection(img_ds.GetProjection())##sets same projection as input
	# outdata.GetRasterBand(1).WriteArray(class_prediction_)
	# outdata.FlushCache() ##saves to disk!!
	# print('Image saved to: {}'.format(classification_image))

	# # validation / accuracy assessment

	# # preparing ttxt file

	# print('------------------------------------', file=open(results_txt, "a"))
	# print('VALIDATION', file=open(results_txt, "a"))

	# # laod training data from shape file
	# shape_dataset_v = ogr.Open(validation)
	# shape_layer_v = shape_dataset_v.GetLayer()
	# mem_drv_v = gdal.GetDriverByName('MEM')
	# mem_raster_v = mem_drv_v.Create('',img_ds.RasterXSize,img_ds.RasterYSize,1,gdal.GDT_UInt16)
	# mem_raster_v.SetProjection(img_ds.GetProjection())
	# mem_raster_v.SetGeoTransform(img_ds.GetGeoTransform())
	# mem_band_v = mem_raster_v.GetRasterBand(1)
	# mem_band_v.Fill(0)
	# mem_band_v.SetNoDataValue(0)

	# # http://gdal.org/gdal__alg_8h.html#adfe5e5d287d6c184aab03acbfa567cb1
	# # http://gis.stackexchange.com/questions/31568/gdal-rasterizelayer-doesnt-burn-all-polygons-to-raster
	# err_v = gdal.RasterizeLayer(mem_raster_v, [1], shape_layer_v, None, None, [1],  [att_,"ALL_TOUCHED=TRUE"])
	# assert err_v == gdal.CE_None

	# roi_v = mem_raster_v.ReadAsArray()



	# # vizualise
	# plt.subplot(221)
	# plt.imshow(img[:, :, 0], cmap=plt.cm.Greys_r)
	# plt.title('RS_Image - first band')

	# plt.subplot(222)
	# plt.imshow(class_prediction, cmap=plt.cm.Spectral)
	# plt.title('Classification result')


	# plt.subplot(223)
	# plt.imshow(roi, cmap=plt.cm.Spectral)
	# plt.title('Training Data')

	# plt.subplot(224)
	# plt.imshow(roi_v, cmap=plt.cm.Spectral)
	# plt.title('Validation Data')

	# plt.show()


	# # Find how many non-zero entries we have -- i.e. how many validation data samples?
	# n_val = (roi_v > 0).sum()
	# print('{n} validation pixels'.format(n=n_val))
	# print('{n} validation pixels'.format(n=n_val), file=open(results_txt, "a"))

	# # What are our validation labels?
	# labels_v = np.unique(roi_v[roi_v > 0])
	# print('validation data include {n} classes: {classes}'.format(n=labels_v.size, classes=labels_v))
	# print('validation data include {n} classes: {classes}'.format(n=labels_v.size, classes=labels_v), file=open(results_txt, "a"))
	# # Subset the classification image with the validation image = X
	# # Mask the classes on the validation dataset = y
	# # These will have n_samples rows
	# X_v = class_prediction[roi_v > 0]
	# y_v = roi_v[roi_v > 0]

	# print('Our X matrix is sized: {sz_v}'.format(sz_v=X_v.shape))
	# print('Our y array is sized: {sz_v}'.format(sz_v=y_v.shape))

	# # Cross-tabulate predictions
	# # confusion matrix
	# convolution_mat = pd.crosstab(y_v, X_v, margins=True)
	# print(convolution_mat)
	# print(convolution_mat, file=open(results_txt, "a"))
	# # if you want to save the confusion matrix as a CSV file:
	# #savename = 'C:\\save\\to\\folder\\conf_matrix_' + str(est) + '.csv'
	# #convolution_mat.to_csv(savename, sep=';', decimal = '.')

	# # information about precision, recall, f1_score, and support:
	# # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
	# #sklearn.metrics.precision_recall_fscore_support
	# target_names = list()
	# for name in range(1,(labels.size)+1):
	#     target_names.append(str(name))
	# sum_mat = classification_report(y_v,X_v,target_names=target_names)
	# print(sum_mat)
	# print(sum_mat, file=open(results_txt, "a"))

	# # Overall Accuracy (OAA)
	# print('OAA = {} %'.format(accuracy_score(y_v,X_v)*100))
	# print('OAA = {} %'.format(accuracy_score(y_v,X_v)*100), file=open(results_txt, "a"))

	# # Plot
	# x_axis_labels = ['Paddy01','Paddy02', 'Paddy03', 'Sugarcane', 'Buildup', 'Palm', 'Eucalyptus', 'Casava', 'Water01', 'Water02', 'Water03']
	# y_axis_labels =  ['Paddy01','Paddy02', 'Paddy03', 'Sugarcane', 'Buildup', 'Palm', 'Eucalyptus', 'Casava', 'Water01', 'Water02', 'Water03']


	# fig = plt.figure(figsize=(10,7))

	# cm = confusion_matrix(y_test,rf.predict(X_test)) #, normalize='all')
	# #cm = cm.astype('float')/cm.sum(axis=0)[:,np.newaxis]
	# #sn.heatmap(cm, annot=True, fmt='g')
	# sn.heatmap(cm, annot=True, fmt='g',  xticklabels=x_axis_labels, yticklabels=y_axis_labels, cmap='YlGnBu') #, vmin=0, vmax=1)
	# plt.title('Classification confusion matrix')
	# plt.xlabel('classes - predicted')
	# plt.ylabel('classes - truth')
	# plt.xticks(rotation=45)
	# plt.show()

	# # Open raw classification result
	# img_ds = gdal.Open(classification_image, 1)  # open image in read-write mode
	# gt = img_ds.GetGeoTransform()
	# proj = img_ds.GetProjection()
	# Band = img_ds.GetRasterBand(1)

	# # print(gdal.Info(img_ds))
	# # yRasSize = img_ds.RasterYSize
	# # xRasSize = img_ds.RasterXSize
	# # print(yRasSize, xRasSize)

	# # copy file to other dir and process generalization
	# fn_new = r"ml_result/rf_class.tif"
	# driver_tiff = gdal.GetDriverByName("GTiff")
	# # copy original raster
	# ds_copy = driver_tiff.CreateCopy(fn_new, img_ds, strict=0)
	# ds_copy = None
	# print("Copy raster done!")

	# # Generalization
	# ds_class = gdal.Open(r"ml_result/rf_class.tif", 1)
	# Band = ds_class.GetRasterBand(1)
	# gdal.SieveFilter(srcBand=Band, maskBand=None, dstBand=Band, threshold=1, connectedness=1, callback=gdal.TermProgress_nocb)
	# ds_class = None
	# Band = None
	# print("Generalization done!")

	# # mask only sugarcane
	# ds_gen = gdal.Open(r"ml_result/rf_class.tif", 1)
	# ds_mask = r"ml_result/rfgen_cane.tif"
	# gt = ds_gen.GetGeoTransform()
	# proj = ds_gen.GetProjection()

	# Band = ds_gen.GetRasterBand(1)
	# array = Band.ReadAsArray()
	# plt.subplot(121)
	# # plt.figure()
	# plt.imshow(array)
	# plt.title("All classes")

	# # Making mask using sugarcane class = 1
	# sel = 4
	# binmask = np.where((array == sel),1,0)
	# # plt.figure()
	# # plt.imshow(binmask)

	# # Masking generalized classification using binmask
	# driver = gdal.GetDriverByName("GTiff")
	# driver.Register()
	# outds = driver.Create(ds_mask, xsize = binmask.shape[1],
	#                       ysize = binmask.shape[0], bands = 1, 
	#                       eType = gdal.GDT_Int16)
	# outds.SetGeoTransform(gt)
	# outds.SetProjection(proj)
	# outband = outds.GetRasterBand(1)
	# outband.WriteArray(binmask)
	# outband.SetNoDataValue(np.nan)
	# outband.FlushCache()

	# # close your datasets and bands!!!
	# outband = None
	# outds = None

	# # display mask result
	# caneImg = gdal.Open(r"ml_result/rfgen_cane.tif")
	# Band = caneImg.GetRasterBand(1)
	# array = Band.ReadAsArray()
	# #plt.figure()
	# plt.subplot(122)
	# plt.imshow(array)
	# plt.title("Mask sugarcane")


	# shp_ds = r"ml_result/rf_sugarcane.shp"
	# src_ds = gdal.Open(r"ml_result/rfgen_cane.tif",1)
	# srcband = src_ds.GetRasterBand(1)
	# dst_layerName = "sugarcane"
	# drv = ogr.GetDriverByName("ESRI Shapefile")

	# if os.path.exists(shp_ds):
	#     drv.DeleteDataSource(shp_ds)

	# dst_ds = drv.CreateDataSource(shp_ds)

	# sp_ref = osr.SpatialReference()
	# sp_ref.SetFromUserInput("EPSG:32648")

	# dst_layer = dst_ds.CreateLayer(dst_layerName, srs=sp_ref)

	# fld = ogr.FieldDefn("VALUE", ogr.OFTInteger)
	# dst_layer.CreateField(fld)
	# dst_field = dst_layer.GetLayerDefn().GetFieldIndex("VALUE")

	# gdal.Polygonize(srcband, None, dst_layer, 0, [], callback=None)
	# dst_ds.Destroy()
	# print('Done!')

	print()
	print(f'Final Time Execution --- %s Seconds ---' % (time.time() - start_time))
	print()
	import datetime
	SecToConvert =  time.time() - start_time #56000
	ConvertedSec = str(datetime.timedelta(seconds = SecToConvert))
	print("Converted Results are: ", ConvertedSec)
	print()



	########################################
	# ------------------------------------ #
	# ------------------------------------ #
	########################################

	########################################
	# ------------------------------------ #
	# ------------------------------------ #
	########################################




	third_stage.info('Already done ...')
		

	with zipfile.ZipFile('shape_file_cls_results.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
		zipdir('ml_result/', zipf)

	with zipfile.ZipFile('shape_file_reg_results.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
		zipdir('regression/shp/', zipf)


	# with open('shape_file_results.zip') as f:
	#    st.download_button('DOWNLOAD SHAPE FILE RESULTS (.ZIP)', f)  # Defaults to 'text/plain'

	# # ---
	# # Binary files

	# binary_contents = b'whatever'

	# # Different ways to use the API

	# third_stage.download_button('Download file', binary_contents)  # Defaults to 'application/octet-stream'

	with open('shape_file_cls_results.zip', 'rb') as f:
	   third_stage.download_button('DOWNLOAD CLS SHAPE FILE RESULTS (.ZIP)', f, file_name='shape_file_cls_results.zip')  # Defaults to 'application/octet-stream'


	with open('shape_file_reg_results.zip', 'rb') as f:
	   third_stage.download_button('DOWNLOAD REG SHAPE FILE RESULTS (.ZIP)', f, file_name='shape_file_reg_results.zip')  # Defaults to 'application/octet-stream'


	# You can also grab the return value of the button,
	# just like with any other button.

	# if third_stage.download_button(...):
	#    third_stage.write('Thanks for downloading!')

	OUTPUT_TEXT = 'Your file is ready to download! with total time spent: '+str( ConvertedSec )

	third_stage.success(OUTPUT_TEXT)


