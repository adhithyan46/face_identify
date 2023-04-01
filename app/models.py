from django.contrib.auth.base_user import AbstractBaseUser
from django.contrib.auth.models import AbstractUser, PermissionsMixin, Permission
from django.db import models
from datetime import datetime
import os

# Create your models here.
sex_choice = (
('Male', 'Male'),
('Female', 'Female')
)


class Login(AbstractUser):
	is_user=models.BooleanField(default=False,unique=False)
	email=models.EmailField(max_length=255,unique=True)
	USERNAME_FIELD ='email'
	REQUIRED_FIELDS = []
	#username = models.CharField(max_length=150, unique=False)
	# is_manager=models.BooleanField(default=False)
# class Login(AbstractUser):
# 	is_admin=models.BooleanField(default=False)
# 	is_Employee=models.BooleanField(default=False)

class Employee(models.Model):
	user = models.OneToOneField(Login, on_delete=models.CASCADE, related_name='user')
	id = models.CharField(primary_key=True, max_length=10)
	name=models.CharField(max_length=50)
	contact_number = models.CharField(max_length=50)
	date_of_birth = models.CharField(max_length=50)
	date_of_joining = models.CharField(max_length=50)
	department = models.CharField(max_length=50)
	designation = models.CharField(max_length=50)
	gender = models.CharField(max_length=50, choices=sex_choice)
	team = models.CharField(max_length=50)
	def __str__(self):
		return self.name


	def num_photos(self):
		try:
			DIR = f"app/facerec/dataset/{self.name}_{self.id}"
			img_count = len(os.listdir(DIR))
			return img_count
		except:
			return 0
# class Manager(models.Model):
# 	user = models.ForeignKey(Login, on_delete=models.CASCADE, related_name='manager')
# 	id = models.CharField(primary_key=True,max_length=10)
# 	name=models.CharField(max_length=50)
# 	email=models.EmailField(max_length=50)
# 	contact_number = models.CharField(max_length=50)
# 	date_of_birth = models.CharField(max_length=50)
# 	date_of_joining = models.CharField(max_length=50)
# 	department = models.CharField(max_length=50)
# 	designation = models.CharField(max_length=50)
# 	gender = models.CharField(max_length=50, choices=sex_choice, default='Male')
# 	team = models.CharField(max_length=50)
# 	def __str__(self):
# 		return self.name


class Detected_in(models.Model):
	emp_id = models.ForeignKey(Employee, on_delete=models.CASCADE, null=True)
	entry = models.DateTimeField()
	photo = models.ImageField(upload_to='detected/', default='app/facerec/detected/noimg.png')

	def __str__(self):
		emp = Employee.objects.get(name=self.emp_id)
		return f"{emp.name} {self.entry}"


class Detected_out(models.Model):
	emp_id = models.ForeignKey(Employee, on_delete=models.CASCADE, null=True)
	out = models.DateTimeField()
	photo = models.ImageField(upload_to='detected/', default='app/facerec/detected/noimg.png')


	def __str__(self):
		emp = Employee.objects.get(name=self.emp_id)
		return f"{emp.name} {self.out}"

class Rep(models.Model):
    emp_id = models.ForeignKey(Employee, on_delete=models.CASCADE, null=True)
    department = models.CharField(max_length=50)
    entry = models.DateTimeField()
    out = models.DateTimeField()
    def __str__(self):
        emp = Employee.objects.get(name=self.emp_id)
        empdep = Employee.objects.get(department=self.department)
        empentry = Detected_in.objects.get(entry = self.entry)
        empout = Detected_out.objects.get(out = self.out)
        return emp.name, empdep.department, empentry.entry, empout.out


# class Hours(models.Model):
# 	emp_id = models.ForeignKey(Employee, on_delete=models.CASCADE, null=True)
# 	total_hours = models.CharField(max_length=50)
#
# 	def __str__(self):
# 		return f"{self.emp_id.name} - {self.total_hours} hours worked"
#
# 	def calculate_hours_worked(self):
# 		attendance_in = Detected_in.objects.filter(emp_id=self.emp_id)
# 		attendance_out = Detected_out.objects.filter(emp_id=self.emp_id)
#
# 		if attendance_in and attendance_out:
# 			total_hours_worked = attendance_out.last().out - attendance_in.last().entry
# 			self.total_hours = total_hours_worked.total_seconds() / 3600
# 			self.save()


class report(models.Model):
    emp_id = models.ForeignKey(Employee, on_delete=models.CASCADE, null=True)
    department = models.CharField(max_length=50)
    entry = models.DateTimeField()
    out = models.DateTimeField()
    def __str__(self):
        emp = Employee.objects.get(name=self.emp_id)
        empdep = Employee.objects.get(department=self.department)
        empentry = Detected_in.objects.get(entry = self.entry)
        empout = Detected_out.objects.get(out = self.out)
        return emp.name, empdep.department, empentry.entry, empout.out



# class CustomUser(AbstractUser):
#     username = None
#     email = models.EmailField(_('email address'), unique=True)
#
#     USERNAME_FIELD = 'email'
#     REQUIRED_FIELDS = []
#
#     objects = CustomUserManager()

class Content(models.Model):
    user= models.ForeignKey(Login, on_delete=models.DO_NOTHING)
    subject = models.CharField(max_length=200)
    content = models.TextField()
   # date = models.DateField()
    reply = models.TextField(null=True, blank=True)






