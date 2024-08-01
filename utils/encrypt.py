from cryptography.fernet import Fernet


def crypt(filepath):
    with open('key.key','rb') as key_file:
        key=key_file.read()
    fernet = Fernet(key)
    with open(filepath, 'rb') as file:
        original = file.read()
    encrypted = fernet.encrypt(original)
    with open(filepath,'wb') as encrypted_file:
        encrypted_file.write(encrypted)
 

def decrypt(filepath):
    with open('key.key','rb') as key_file:
        key=key_file.read()
    fernet = Fernet(key)
    with open(filepath, 'rb') as file:
        encrypted = file.read().strip()
        decrypted=fernet.decrypt(encrypted)
    with open(filepath, 'wb') as decrypted_file:
        decrypted_file.write(decrypted)

if __name__ == '__main__':
    crypt('ciao.txt')
    decrypt('ciao.txt')