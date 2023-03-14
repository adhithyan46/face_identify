from django.contrib import admin
from .models import Employee, Detected_in, Detected_out, Rep

# Register your models here.

admin.site.register(Employee)
admin.site.register(Detected_in)
admin.site.register(Detected_out)
admin.site.register(Rep)