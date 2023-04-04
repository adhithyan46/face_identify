import math
from urllib import request

from django.contrib.auth import authenticate, logout
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render, get_object_or_404
from django.http import HttpResponseRedirect, HttpResponse
from django.urls import reverse
from django.utils import timezone
from django.db.models import Q


#import accounts
from face_rec_django.settings import LOGIN_REDIRECT_URL
from .facerec.faster_video_stream import stream
from .facerec.click_photos import click
from .facerec.train_faces import trainer
from .models import Detected_out, Employee, Detected_in, Rep, report, Login, Content
from .forms import EmployeeForm, LoginRegister, ContentForm
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
#                 return redirect('report2')
#         else:
#             return HttpResponse("Username or Password is incorrect!!!")
#
#     return render (request,'index.html')




#@login_required(login_url='/app/login/')
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

#@login_required(login_url='/app/login/')
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



#@login_required(login_url='/app/login/')
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


#@login_required(login_url='/app/login/')
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



#@login_required(login_url='/app/login/')
def index(request):
    return render(request, 'app/index.html')

#@login_required(login_url='/app/login/')
def video_stream(request):
    stream()
    return HttpResponseRedirect(reverse('index'))

#@login_required(login_url='/app/login/')
def add_photos(request):
	emp_list = Employee.objects.all()
	return render(request, 'app/add_photos.html', {'emp_list': emp_list})

#@login_required(login_url='/app/login/')
def click_photos(request, emp_id):
	cam = cv2.VideoCapture(0)
	emp = get_object_or_404(Employee, id=emp_id)
	click(emp.name, emp.id, cam)
	return HttpResponseRedirect(reverse('add_photos'))

#@login_required(login_url='/accounts/login/')
def train_model(request):
	trainer()
	return HttpResponseRedirect(reverse('index'))

#@login_required(login_url='/app/login/')
def detected(request):
	if request.method == 'GET':
		date_formatted = datetime.datetime.today().date()
		date = request.GET.get('search_box', None)
		if date is not None:
			date_formatted = datetime.datetime.strptime(date, "%Y-%m-%d").date()
		det_list = Detected_in.objects.filter(entry__date=date_formatted).order_by('entry').reverse()

	# det_list = Detected.objects.all().order_by('time_stamp').reverse()
	return render(request, 'app/detected.html', {'det_list': det_list, 'date': date_formatted})
#@login_required(login_url='/app/login/')
def detected_out(request):
	if request.method == 'GET':
		date_formatted = datetime.datetime.today().date()
		date = request.GET.get('search_box', None)
		if date is not None:
			date_formatted = datetime.datetime.strptime(date, "%Y-%m-%d").date()
		det_list = Detected_out.objects.filter(out__date=date_formatted).order_by('out').reverse()

	# det_list = Detected.objects.all().order_by('time_stamp').reverse()
	return render(request, 'app/detectedout.html', {'det_list': det_list, 'date': date_formatted})
#@login_required(login_url='/accounts/login/')
def identify(request):
    video_capture1 = cv2.VideoCapture(0)
    video_capture2 = cv2.VideoCapture(1)
    identify_faces(video_capture1, video_capture2)
    return HttpResponseRedirect(reverse('index'))

#@login_required(login_url='/accounts/login/')
# def add_emp(request):
#     if request.method == "POST":
#         form = EmployeeForm(request.POST)
#         if form.is_valid():
#             emp = form.save()
#             # post.author = request.user
#             # post.published_date = timezone.now()
#             # post.save()
#             return HttpResponseRedirect(reverse('index'))
#     else:
#         form = EmployeeForm()
#     return render(request, 'app/add_emp.html', {'form': form})



#@login_required(login_url='/app/registration/login/')
# def logout(request):
#     logout(request)
#     return redirect('LoginPage')
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

#@login_required(login_url='/app/login/')
def admin(request):
    return redirect(request,'admin:index')

#@login_required(login_url='/app/login/')
def attendece_rep(request):
    if request.method == 'GET':
        date_formatted = datetime.datetime.today().date()
        date = request.GET.get('search_box', None)
        if date is not None:
            date_formatted = datetime.datetime.strptime(date, "%Y-%m-%d").date()

        det_list_out = Detected_out.objects.filter(out__date=date_formatted).order_by('emp_id_id').reverse()
        det_list_in = Detected_in.objects.filter(entry__date=date_formatted).order_by('emp_id_id').reverse()
        report = Rep.objects.filter(entry__date=date_formatted).order_by('emp_id_id').reverse()

        attendance_data=[]

        for det1, det2 in zip(det_list_out, det_list_in):
            if det1.emp_id_id == det2.emp_id_id:
                total_time = det1.out - det2.entry
                total_time_str = str(datetime.timedelta(seconds=total_time.seconds))
                attendance_data.append({
                    'employee_id': det1.emp_id_id,
                    'employee_name': det1.emp_id,
                    'entry_time': det2.entry,
                    'exit_time': det1.out,
                    'total_time': total_time_str,
                })



        context = {
            'attendance_data':attendance_data,
            'date': date_formatted,
            'report': report,
        }

    return render(request, 'app/attendencereport.html', context)



#@login_required(login_url='/app/login/')
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
        attendance_data = []
        report_data=[]

        for det1, det2 in zip(det_list_out, det_list_in):
            if det1.emp_id_id == det2.emp_id_id:
                total_time = det1.out - det2.entry
                total_time_str = str(datetime.timedelta(seconds=total_time.seconds))
                attendance_data.append({
                    'employee_id': det1.emp_id_id,
                    'employee_name': det1.emp_id,
                    'entry_time': det2.entry,
                    'exit_time': det1.out,
                    'total_time': total_time_str,
                })
        for rep1,rep2 in zip(report_in,report_out):
            if rep1.emp_id_id ==rep2.emp_id_id:
                total_time= rep2.out-rep1.entry
                total_time_str=str(datetime.timedelta(seconds=total_time.seconds))
                report_data.append({
                    'employee_id': rep1.emp_id_id,
                    'employee_name': rep1.emp_id,
                    'entry_time': rep1.entry,
                    'exit_time': rep2.out,
                    'total_time': total_time_str,

                })
        context = {
            'attendance_data':attendance_data,
            'report_data':report_data,
            'date': date_formatted,
            'report1': report1,


        }

    return render(request, 'app/report.html', context)





#@login_required(login_url='/app/login/')
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




from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
# from django.shortcuts import render, redirect

# def login_view(request):
#     if request.method == 'POST':
#         username = request.POST['username']
#         password = request.POST['password']
#         user = authenticate(request, username=username, password=password)
#         if user is not None:
#             login(request, user)
#             request.session['user_id'] = user.id
#             return redirect('home')
#         else:
#             error_msg = 'Invalid login credentials.'
#     else:
#         error_msg = ''
#     return render(request, 'login.html', {'error_msg': error_msg})
#@login_required(login_url='/app/login/')
def logout_view(request):
    logout(request)
    return redirect('login_view')


from django.contrib.auth import authenticate, login
#@login_required(login_url='/app/login/')
def login_view(request):
    if request.method == 'POST':
        email = request.POST.get('user')
        password = request.POST.get('password')
        user = authenticate(request, email=email,password=password)
        if user is not None:
            login(request, user)
            if user.is_user:
                return redirect('home')
            elif user.is_staff:
                return redirect('index')
            else:
                return redirect('index')
        else:
            messages.info(request,'Invalid Credentials')
    return render(request,'app/login.html')
# def login_view(request):
#     if request.method == 'POST':
#         email = request.POST.get('email') # use 'email' instead of 'uname'
#         password = request.POST.get('Password')
#         try:
#             user = Login.objects.get(email=email, is_user=True)
#         except Login.DoesNotExist:
#             user = None
#         if user is not None and user.check_password(password):
#             login(request, user)
#             if user.is_staff:
#                 return redirect('index')
#             else:
#                 return redirect('index')
#         else:
#             messages.info(request,'Invalid Credentials')
#     return render(request,'app/login.html')


# # @login_required(login_url='/app/login/')
# def manager_add(request):
#     form1=LoginRegister()
#     form2=ManagerForm()
#     if request.method=='POST':
#         form1=LoginRegister(request.POST)
#         form2=EmployeeForm(request.POST)
#         if form1.is_valid() and form2.is_valid():
#             a=form1.save(commit=False)
#             a.is_manager=True
#             a.save()
#             user1=form2.save(commit=False)
#             user1.user=a
#             user1.save()
#             messages.info(request,'User Registered Successfully')
#             return redirect('index')
#     return render(request,'app/employee_add.html',{'form1':form1,'form2':form2})

# @login_required(login_url='/app/login/')
def home(request):
    return render(request, 'app/home.html')

def add_emp(request):
    form1=LoginRegister()
    form2=EmployeeForm()
    if request.method=='POST':
        form1=LoginRegister(request.POST)
        form2=EmployeeForm(request.POST)

        if form1.is_valid() and form2.is_valid():
            a=form1.save(commit=False)
            a.is_user=True
            a.email=form1.cleaned_data['email']
            a.save()
            user1=form2.save(commit=False)
            user1.user=a
            user1.save()
            return redirect('index')
    return render(request,'app/add_emp.html',{'form1':form1,'form2':form2})



def Content_add(request):
    form = ContentForm()
    u = request.user
    if request.method == 'POST':
        form = ContentForm(request.POST)
        if form.is_valid():
            obj = form.save(commit=False)
            obj.user = u
            obj.save()
            messages.info(request, 'Content Registered Successfully')
            return redirect('Contentt')
    else:
        form = ContentForm()
    return render(request, 'app/content_add.html', {'form': form})


def Contentt(request):
    n = Content.objects.filter(user=request.user)
    return render(request, 'app/content.html', {'Content': n})

def Content_admin(request):
    n = Content.objects.all()
    return render(request, 'admintemp/content.html', {'Content': n})
def reply_Content(request,id):
    content = Content.objects.get(id=id)
    if request.method == 'POST':
        r = request.POST.get('reply')
        content.reply = r
        content.save()
        messages.info(request, 'Reply send for content')
        return redirect('Content_admin')
    return render(request, 'admintemp/content_reply.html ',{'content': content})