# Generated by Django 4.0.5 on 2023-03-24 09:34

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0011_login_remove_employee_password_delete_hours'),
    ]

    operations = [
        migrations.RenameField(
            model_name='employee',
            old_name='name',
            new_name='user',
        ),
    ]
