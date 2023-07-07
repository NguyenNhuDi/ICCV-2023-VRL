class Animal:

    def __init__(self, sound):
        self.sound = sound

    def get_voice(self):
        return self.sound

    def talk(self):
        print(self.get_voice())


class Cow(Animal):
    def get_voice(self):
        return 'hello world'


if __name__ == '__main__':
    animal = Animal('animal sounds')
    cow = Cow('moo')

    animal.talk()
    cow.talk()
    