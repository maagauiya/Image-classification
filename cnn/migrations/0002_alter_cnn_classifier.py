# Generated by Django 4.0.3 on 2022-03-03 10:56

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('cnn', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='cnn',
            name='classifier',
            field=models.TextField(),
        ),
    ]
