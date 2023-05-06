import django_filters
from django import forms
from django_filters import CharFilter, filters

from app.models import Employee


class EmployeeFilter(django_filters.FilterSet):
    id = CharFilter(field_name='id', label="", lookup_expr='icontains', widget=forms.TextInput(attrs={
                  'placeholder': 'Search Name', 'class': 'form-control'}))

    class Meta:
        model = Employee
        fields = ('id',)
