from django.db import models
from phonenumber_field.modelfields import PhoneNumberField
from django.contrib.auth.models import AbstractUser


class User(AbstractUser) :
    phone_number = PhoneNumberField(unique=True)

class Image(models.Model) :
    file = models.ImageField(upload_to='images/', null=True, blank=True)

    def __str__(self):
        return f"{self.file}"