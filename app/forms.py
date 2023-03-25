
from django import forms
from django.contrib.auth.forms import UserCreationForm

from .models import Employee


class EmployeeForm(forms.ModelForm):

    class Meta:
        model = Employee

        fields = ('id', 'name', 'contact_number', 'date_of_birth', 'date_of_joining', 'department', 'designation', 'gender','team')
#
# class LoginRegister(UserCreationForm):
#     username=forms.CharField()
#     password1=forms.CharField(label='password',widget=forms.PasswordInput)
#     Password2=forms.CharField(label='confirm password',widget=forms.PasswordInput)
#     class Meta:
#         model=Login
#         fields=('username','password1','password2',)
# class EmployeeRegister(forms.ModelForm):
#
#
#     class Meta:
#         model=Employee
#         fields='__all__'
#         exclude=('user',)



