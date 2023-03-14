from django.shortcuts import render, get_object_or_404
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.utils import timezone
from .facerec.faster_video_stream import stream
from .facerec.click_photos import click
from .facerec.train_faces import trainer
from .models import Detected_out, Employee, Detected_in
from .forms import EmployeeForm
import cv2
import pickle
import face_recognition
import datetime
from cachetools import TTLCache


cache = TTLCache(maxsize=20, ttl=60)


def identify1(frame, name, buf, buf_length, known_conf):

    if name in cache:
        return
    count = 0
    for ele in buf:
        count += ele.count(name)
    
    if count >= known_conf:
        entry = datetime.datetime.now(tz=timezone.utc)
        print(name, entry)
        cache[name] = 'detected'
        path = 'detected/{}_{}.jpg'.format(name, entry)
        write_path = 'media/' + path
        cv2.imwrite(write_path, frame)
        try:
            emp = Employee.objects.get(name=name)
            emp.detected_in_set.create(entry=entry, photo=path)
        except:
            pass 	        


def identify2(frame, name, buf, buf_length, known_conf):

    if name in cache:
        return
    count = 0
    for ele in buf:
        count += ele.count(name)
    
    if count >= known_conf:
        out = datetime.datetime.now(tz=timezone.utc)
        print(name, out)
        cache[name] = 'detected'
        path = 'detected/{}_{}.jpg'.format(name, out)
        write_path = 'media/' + path
        cv2.imwrite(write_path, frame)
        try:
            emp = Employee.objects.get(name=name)
            emp.detected_out_set.create(out = out, photo=path)
        except:
            pass 	        




def predict(rgb_frame, knn_clf=None, model_path=None, distance_threshold=0.5):

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    # X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=2)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    # print(closest_distances)
    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]



def identify_faces(video_capture):

    buf_length = 10
    known_conf = 6
    buf = [[]] * buf_length
    i = 0

    process_this_frame = True

    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame = small_frame[:, :, ::-1]

        if process_this_frame:
            predictions = predict(rgb_frame, model_path="app/facerec/models/trained_model.clf")
        process_this_frame = not process_this_frame
        face_names = []
        for name, (top, right, bottom, left) in predictions:
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
            identify1(frame, name, buf, buf_length, known_conf)
            face_names.append(name)
        buf[i] = face_names
        i = (i + 1) % buf_length


                       #IDENTIFY FACES OUT



def identify_faces_out(video_capture):

    buf_length = 10
    known_conf = 6
    buf = [[]] * buf_length
    i = 0
    process_this_frame = True
    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame = small_frame[:, :, ::-1]
        if process_this_frame:
            predictions = predict(rgb_frame, model_path="app/facerec/models/trained_model.clf")
        process_this_frame = not process_this_frame
        face_names = []
        for name, (top, right, bottom, left) in predictions:
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            identify2(frame, name, buf, buf_length, known_conf)
            face_names.append(name)
        buf[i] = face_names
        i = (i + 1) % buf_length


        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()


def index(request):
    return render(request, 'app/index.html')


def video_stream(request):
    stream()
    return HttpResponseRedirect(reverse('index'))


def add_photos(request):
	emp_list = Employee.objects.all()
	return render(request, 'app/add_photos.html', {'emp_list': emp_list})


def click_photos(request, emp_id):
	cam = cv2.VideoCapture(0)
	emp = get_object_or_404(Employee, id=emp_id)
	click(emp.name, emp.id, cam)
	return HttpResponseRedirect(reverse('add_photos'))


def train_model(request):
	trainer()
	return HttpResponseRedirect(reverse('index'))


def detected(request):
	if request.method == 'GET':
		date_formatted = datetime.datetime.today().date()
		date = request.GET.get('search_box', None)
		if date is not None:
			date_formatted = datetime.datetime.strptime(date, "%Y-%m-%d").date()
		det_list = Detected_in.objects.filter(entry__date=date_formatted).order_by('entry').reverse()

	# det_list = Detected.objects.all().order_by('time_stamp').reverse()
	return render(request, 'app/detected.html', {'det_list': det_list, 'date': date_formatted})

def detected_out(request):
	if request.method == 'GET':
		date_formatted = datetime.datetime.today().date()
		date = request.GET.get('search_box', None)
		if date is not None:
			date_formatted = datetime.datetime.strptime(date, "%Y-%m-%d").date()
		det_list = Detected_out.objects.filter(out__date=date_formatted).order_by('out').reverse()

	# det_list = Detected.objects.all().order_by('time_stamp').reverse()
	return render(request, 'app/detectedout.html', {'det_list': det_list, 'date': date_formatted})

def identify(request):
    video_capture = cv2.VideoCapture(0)
    identify_faces(video_capture)
    return HttpResponseRedirect(reverse('index'))

def identify_out(request):
    video_capture = cv2.VideoCapture(1)
    identify_faces_out(video_capture)
    return HttpResponseRedirect(reverse('index'))


def add_emp(request):
    if request.method == "POST":
        form = EmployeeForm(request.POST)
        if form.is_valid():
            emp = form.save()
            # post.author = request.user
            # post.published_date = timezone.now()
            # post.save()
            return HttpResponseRedirect(reverse('index'))
    else:
        form = EmployeeForm()
    return render(request, 'app/add_emp.html', {'form': form})



def identify_faces(video_capture1, video_capture2):

    buf_length = 10
    known_conf = 6
    buf = [[]] * buf_length
    i = 0
    buf_length1 = 10
    known_conf1 = 6
    buf1 = [[]] * buf_length1
    i1 = 0

    process_this_frame = True

    while True:
        # Grab a single frame of video
        ret, frame = video_capture1.read()
        ret1, frame1 = video_capture2.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        small_frame1 = cv2.resize(frame1, (0, 0), fx=0.25, fy=0.25)


        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = small_frame[:, :, ::-1]
        rgb_frame1 = small_frame1[:, :, ::-1]

        if process_this_frame:
            predictions = predict(rgb_frame, model_path="app/facerec/models/trained_model.clf")
            predictions1 = predict(rgb_frame1, model_path="app/facerec/models/trained_model.clf")
            # print(predictions)

        process_this_frame = not process_this_frame

        face_names = []
        face_names1 = []

        for(name, (top, right, bottom, left))  in predictions and (name, (top, right, bottom, left)) in predictions1:

            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame1, (left, top), (right, bottom), (0, 0, 255), 2)

           

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            
             # Draw a label with a name below the face           
            cv2.rectangle(frame1, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font1 = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame1, name, (left + 6, bottom - 6), font1, 1.0, (255, 255, 255), 1)

          

            identify1(frame, name, buf, buf_length, known_conf)
            identify2(frame1, name, buf1, buf_length1, known_conf1)
            

            face_names.append(name)
            face_names1.append(name) 
           
            
           

        buf[i] = face_names
        buf1[i1] = face_names1
        i = (i + 1) % buf_length
        i1 = (i1 + 1) % buf_length1


        # print(buf)


        # Display the resulting image
        cv2.imshow('Video', frame)
        cv2.imshow('Video1', frame1)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture1.release()
    video_capture2.release()
    cv2.destroyAllWindows()