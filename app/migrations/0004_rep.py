# Generated by Django 4.1.5 on 2023-02-16 11:42

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0003_detected_in_photo_detected_out_photo'),
    ]

    operations = [
        migrations.CreateModel(
            name='Rep',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=50)),
                ('department', models.CharField(max_length=50)),
                ('entry', models.DateTimeField()),
                ('out', models.DateTimeField()),
                ('emp_id', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='app.employee')),
            ],
        ),
    ]