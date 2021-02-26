import functools


user = {
    'name': 'Boby',
    'access':'admin'
}


def admin_required(func):

    @functools.wraps(func)
    def is_required(*args, **kwargs):
        if user['access'] == 'admin':
            return func(*args, **kwargs)

    return is_required
        



@admin_required
def the_admin():
    return 'THIS IS ADMIN'


@admin_required
def the_user():
    return 'THIS IS USER'




print(the_admin('admin'))
# print(the_user('user'))