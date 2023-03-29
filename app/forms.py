
from django import forms
from django.contrib.auth.forms import UserCreationForm

#from django.contrib.auth.forms import UserCreationForm

from .models import Employee, Login


# class EmployeeForm(forms.ModelForm):
#
#     class Meta:
#         model = Employee
#
#         fields = ('id','name','password','email','contact_number','date_of_birth','date_of_joining','department','designation','gender','team')
# #
class LoginRegister(UserCreationForm):
    user=forms.CharField()
    password1=forms.CharField(label='password',widget=forms.PasswordInput)
    #Password2=forms.CharField(label='confirm password',widget=forms.PasswordInput)
    class Meta:
        model=Login
        fields=('user','password1')
class EmployeeForm(forms.ModelForm):
    class Meta:
        model=Employee
        fields='__all__'
        exclude=('user',)

# from django.contrib.auth import get_user_model
# from django.contrib.auth.forms import UserCreationForm
#
#
# class UserAdminCreationForm(UserCreationForm):
#     """
#     A Custom form for creating new users.
#     """
#
#     class Meta:
#         model = get_user_model()
#         fields = ['email']

