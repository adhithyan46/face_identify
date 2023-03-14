# Generated by Django 4.1.5 on 2023-02-01 12:23

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Employee',
            fields=[
                ('id', models.CharField(max_length=10, primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=50)),
                ('contact_number', models.CharField(max_length=50)),
                ('date_of_birth', models.CharField(max_length=50)),
                ('date_of_joining', models.CharField(max_length=50)),
                ('department', models.CharField(max_length=50)),
                ('designation', models.CharField(max_length=50)),
                ('gender', models.CharField(choices=[('Male', 'Male'), ('Female', 'Female')], default='Male', max_length=50)),
                ('team', models.CharField(max_length=50)),
            ],
        ),
        migrations.CreateModel(
            name='Detected',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('entry', models.DateTimeField()),
                ('emp_id', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='app.employee')),
            ],
        ),
    ]
