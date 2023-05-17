import datetime

from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.forms import DateInput

#from django.contrib.auth.forms import UserCreationForm

from .models import Employee, Login, Content


# class EmployeeForm(forms.ModelForm):
#
#     class Meta:
#         model = Employee
#
#         fields = ('id','name','password','email','contact_number','date_of_birth','date_of_joining','department','designation','gender','team')
# #
class LoginRegister(UserCreationForm):
    email = forms.CharField(max_length=30)
    # Password1=forms.CharField(label='password',widget=forms.PasswordInput)
    # Password2=forms.CharField(label='confirm password',widget=forms.PasswordInput)
    class Meta:
        model=Login
        fields=('email','password1','password2')
class EmployeeForm(forms.ModelForm):
    class Meta:
        model=Employee
        fields='__all__'
        exclude=('user',)

        def __init__(self, *args, **kwargs):
            super(EmployeeForm, self).__init__(*args, **kwargs)
            self.fields['user'].queryset = Login.objects.filter(is_user=True).exclude(user__isnull=False)



class ContentForm(forms.ModelForm):
    #date = forms.DateField(widget=DateInput,input_formats=['%m/%d%y'])

    class Meta:
        model = Content
        fields = ('subject', 'content')









