import math
from urllib import request

from django.contrib.auth import authenticate
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render, get_object_or_404
from django.http import HttpResponseRedirect, HttpResponse
from django.urls import reverse
from django.utils import timezone
from django.db.models import Q

from accounts.forms import UserAdminCreationForm
#import accounts
from face_rec_django.settings import LOGIN_REDIRECT_URL
from .facerec.faster_video_stream import stream
from .facerec.click_photos import click
from .facerec.train_faces import trainer
from .models import Detected_out, Employee, Detected_in, Rep, report
from .forms import EmployeeForm
import cv2
import pickle
import face_recognition
import datetime
from cachetools import TTLCache
from django.contrib import messages
from django.http import FileResponse
import io
#from reportlab.pdfgen import canvas
#from reportlab.lib.units import inch
#from reportlab.lib.pagesizes import letter


cache = TTLCache(maxsize=20, ttl=60)
cache1 = TTLCache(maxsize=20, ttl=60)



# def LoginPage(request):
#     if request.method=='POST':
#         username=request.POST.get('username')
#         password=request.POST.get('pass')
#         user=authenticate(request,username=username,password=password)
#         if user is not None:
#             login(request,user)
#             if user.is_admin:
#                 return redirect('index')
#             else:
#                 return redirect('index')
#         else:
#             return HttpResponse ("Username or Password is incorrect!!!")
#
#     return render (request,'index.html')
#



@login_required(login_url='/accounts/login/')
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

@login_required(login_url='/accounts/login/')
def identify2(frame1, name1, buf1, buf_length1, known_conf1):

    if name1 in cache1:
        return
    count = 0
    for ele in buf1:
        count += ele.count(name1)
    
    if count >= known_conf1:
        out = datetime.datetime.now(tz=timezone.utc)
        print(name1, out)
        cache1[name1] = 'detected'
        path = 'detected/{}_{}.jpg'.format(name1, out)
        write_path = 'media/' + path
        cv2.imwrite(write_path, frame1)
        try:
            emp = Employee.objects.get(name=name1)
            emp.detected_out_set.create(out = out, photo=path)
        except:
            pass 	        



@login_required(login_url='/accounts/login/')
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


@login_required(login_url='/accounts/login/')
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

        for name, (top, right, bottom, left) in predictions:

            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
           

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
          

            identify1(frame, name, buf, buf_length, known_conf)
            

            face_names.append(name)
           
            
        for name1, (top, right, bottom, left) in predictions1:

            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame1, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face           
            cv2.rectangle(frame1, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font1 = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame1, name1, (left + 6, bottom - 6), font1, 1.0, (255, 255, 255), 1)

            identify2(frame1, name1, buf1, buf_length1, known_conf1)

            face_names1.append(name1)    

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



@login_required(login_url='/accounts/login/')
def index(request):
    return render(request, 'app/index.html')

@login_required(login_url='/accounts/login/')
def video_stream(request):
    stream()
    return HttpResponseRedirect(reverse('index'))

@login_required(login_url='/accounts/login/')
def add_photos(request):
	emp_list = Employee.objects.all()
	return render(request, 'app/add_photos.html', {'emp_list': emp_list})

@login_required(login_url='/accounts/login/')
def click_photos(request, emp_id):
	cam = cv2.VideoCapture(0)
	emp = get_object_or_404(Employee, id=emp_id)
	click(emp.name, emp.id, cam)
	return HttpResponseRedirect(reverse('add_photos'))

@login_required(login_url='/accounts/login/')
def train_model(request):
	trainer()
	return HttpResponseRedirect(reverse('index'))

@login_required(login_url='/accounts/login/')
def detected(request):
	if request.method == 'GET':
		date_formatted = datetime.datetime.today().date()
		date = request.GET.get('search_box', None)
		if date is not None:
			date_formatted = datetime.datetime.strptime(date, "%Y-%m-%d").date()
		det_list = Detected_in.objects.filter(entry__date=date_formatted).order_by('entry').reverse()

	# det_list = Detected.objects.all().order_by('time_stamp').reverse()
	return render(request, 'app/detected.html', {'det_list': det_list, 'date': date_formatted})
@login_required(login_url='/accounts/login/')
def detected_out(request):
	if request.method == 'GET':
		date_formatted = datetime.datetime.today().date()
		date = request.GET.get('search_box', None)
		if date is not None:
			date_formatted = datetime.datetime.strptime(date, "%Y-%m-%d").date()
		det_list = Detected_out.objects.filter(out__date=date_formatted).order_by('out').reverse()

	# det_list = Detected.objects.all().order_by('time_stamp').reverse()
	return render(request, 'app/detectedout.html', {'det_list': det_list, 'date': date_formatted})
@login_required(login_url='/accounts/login/')
def identify(request):
    video_capture1 = cv2.VideoCapture(0)
    video_capture2 = cv2.VideoCapture(1)
    identify_faces(video_capture1, video_capture2)
    return HttpResponseRedirect(reverse('index'))

@login_required(login_url='/accounts/login/')
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

# def register(request):
#     if request.method == "POST":
#         form = EmployeeForm(request.POST)
#         if form.is_valid():
#             emp = form.save()
#             # post.author = request.user
#             # post.published_date = timezone.now()
#             # post.save()
#             return HttpResponseRedirect(reverse('index'))
#     else:
#         form = registrationForm()
#     return render(request, 'app/register.html', {'form': form})
    # if request.method == 'POST':
    #     id = request.POST.get('id')
    #     name = request.POST.get('name')
    #     contact_number = request.POST.get('contact_number')
    #     date_of_birth = request.POST.get('date_of_birth')
    #     date_of_joining = request.POST.get('date_of_joining')
    #     department = request.POST.get('department')
    #     designation = request.POST.get('designation')
    #     gender = request.POST.get('gender')
    #     team = request.POST.get('team')
    #     password = request.POST.get('password')

        # Create a new registration instance and save it to the database
        # registration.objects.create(
        #     id=id,
        #     name=name,
        #     contact_number=contact_number,
        #     date_of_birth=date_of_birth,
        #     date_of_joining=date_of_joining,
        #     department=department,
        #     designation=designation,
        #     gender=gender,
        #     team=team,
        #     password=password,
        # )
        #
        # messages.success(request, 'Registration successful')
        # return redirect('register')



#@login_required(login_url='/app/registration/login/')
def logout(request):
    logout(request)
    return redirect('Loginpage')
# def report(request):
#     if request.method == 'GET':
#         date_formatted = datetime.datetime.today().date()
#         date = request.GET.get('search_box', None)
#         if date is not None:
#             date_formatted = datetime.datetime.strptime(date, "%Y-%m-%d").date()
#         det_list_out = Detected_out.objects.filter(out__date=date_formatted).order_by('emp_id_id').reverse()
#         det_list_in = Detected_in.objects.filter(entry__date=date_formatted).order_by('emp_id_id').reverse()
#         report = Rep.objects.filter(entry__date=date_formatted).order_by('emp_id_id').reverse()
#         report_in = Detected_in.objects.all()
#         report_out = Detected_out.objects.all()


def admin(request):
    return redirect('admin:index')

@login_required(login_url='/accounts/login/')
def attendece_rep(request):
    if request.method == 'GET':
        date_formatted = datetime.datetime.today().date()
        date = request.GET.get('search_box', None)
        if date is not None:
            date_formatted = datetime.datetime.strptime(date, "%Y-%m-%d").date()

        det_list_out = Detected_out.objects.filter(out__date=date_formatted).order_by('emp_id_id').reverse()
        det_list_in = Detected_in.objects.filter(entry__date=date_formatted).order_by('emp_id_id').reverse()
        report = Rep.objects.filter(entry__date=date_formatted).order_by('emp_id_id').reverse()

        context = {
            'det_list_out': det_list_out,
            'det_list_in': det_list_in,
            'date': date_formatted,
            'report': report,
        }

    return render(request, 'app/attendencereport.html', context)


"""def report(request):
    if request.method == 'GET':
        date_formatted = datetime.datetime.today().date()
        date = request.GET.get('search_box', None)
        if date is not None:
            date_formatted = datetime.datetime.strptime(date, "%Y-%m-%d").date()
        det_list_out = Detected_out.objects.filter(out__date=date_formatted).order_by('emp_id_id').reverse()
        det_list_in = Detected_in.objects.filter(entry__date=date_formatted).order_by('emp_id_id').reverse()
        report = Rep.objects.filter(entry__date=date_formatted).order_by('emp_id_id').reverse()
        report_in = Detected_in.objects.all()
        report_out = Detected_out.objects.all()
    return render(request, 'app/report.html', {'det_list_out': det_list_out,'det_list_in': det_list_in,'date': date_formatted, 'report' : report, 'report_in':report_in, 'report_out':report_out })"""


@login_required(login_url='/accounts/login/')
def reportt(request):
    if request.method == 'GET':
        date_formatted = datetime.datetime.today().date()
        date = request.GET.get('search_box', None)
        if date is not None:
            date_formatted = datetime.datetime.strptime(date, "%Y-%m-%d").date()
        det_list_out = Detected_out.objects.filter(out__date=date_formatted).order_by('emp_id_id').reverse()
        det_list_in = Detected_in.objects.filter(entry__date=date_formatted).order_by('emp_id_id').reverse()
        report1 = report.objects.filter(entry__date=date_formatted).order_by('emp_id_id').reverse()
        report_in = Detected_in.objects.all()
        report_out = Detected_out.objects.all()

        # #calculate total hours spent for each employee
        # total_hours = {}
        # for det_in in report_in:
        #     emp_id = det_in.emp_id_id
        #     if emp_id not in total_hours:
        #         total_hours[emp_id] = 0
        #     report_out = Detected_out.objects.filter(emp_id_id=emp_id, out__date=date_formatted).first()
        #     if report_out:
        #         total_hours[emp_id] += (report_out.out - det_in.entry).total_seconds() / 3600
        context = {
            'det_list_out': det_list_out,
            'det_list_in': det_list_in,
            'date': date_formatted,
            'report1': report1,
            'report_in':report_in,
            'report_out':report_out,
            #'total_hours':total_hours,

        }

    return render(request, 'app/report.html', context)



# def person(request):
#     if request.method=='POST':
#         name=request.POST['Employee']
#         detected_in=request.POST['Detected_ins']
#         detected_out=request.POST['Detected_outs']
#         report=request.POST['Reps']
#         emps=Employee.object.all()
#         context={
#             'emps':emps
#         }
#         return render(request,'app/person.html')
#     elif request.method=='GET':
#         return render(request,'app/person.html')
#     else:
#         return HttpResponse("An exception occured")

@login_required(login_url='/accounts/login/')
def person(request):
    if request.method == 'POST':
        date_formatted = datetime.datetime.today().date()
        date = request.GET.get('search_box', None)
        if date is not None:
            date_formatted = datetime.datetime.strptime(date, "%Y-%m-%d").date()
        det_list_out = Detected_out.objects.filter(out__date=date_formatted).order_by('emp_id_id').reverse()
        det_list_in = Detected_in.objects.filter(entry__date=date_formatted).order_by('emp_id_id').reverse()
        report = Rep.objects.filter(entry__date=date_formatted).order_by('emp_id_id').reverse()
        report_in = Detected_in.objects.all()
        report_out = Detected_out.objects.all()

        context = {
            'det_list_out': det_list_out,
            'det_list_in': det_list_in,
            'date': date_formatted,
            'report': report,

        }

    return render(request, 'app/report.html',
                  {'det_list_out': det_list_out, 'det_list_in': det_list_in, 'date': date_formatted, 'report': report,
                   'report_in': report_in, 'report_out': report_out })




# def add_emp(request):
#     form1=LoginRegister()
#     form2=EmployeeRegister()
#     if request.method=='POST':
#         form1=LoginRegister(request.POST)
#         form2=EmployeeRegister(request.POST,request.FILES)
#         if form1.is_valid() and form2.is_valid():
#             a=form1.save(commit=False)
#             a.is_admin=True
#             a.save()
#             user1=form2.save(commit=False)
#             user1.user=a
#             user1.save()
#             return redirect('')
#
#     return render(request,'registration.html',{'form1':form1,'form2':form2})



# def loginn(request):
#     if request.method == 'POST':
#         username = request.POST.get('uname')
#         password = request.POST.get('pass')
#         user = authenticate(request, username=username, password=password)
#         if user is not None:
#             login(request, user)
#             if user.is_admin:
#                 return redirect('index')
#             elif user.is_employee:
#                 return redirect('index')
#
#         else:
#             messages.info(request, 'Invalid Credentials')
#     return render(request, 'accounts/login.html')
@login_required(login_url='/accounts/login/')
def register(request):
    form = UserAdminCreationForm()
    if request.method == 'POST':
        form = UserAdminCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('register')
    return render(request, 'register.html', {'form': form})