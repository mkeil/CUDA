struct listElement

{
    int value;
    struct listElement* next;
};

typedef struct listElement listElement;

listElement* newElement (int newValue);

void delElement (listElement* element);

void printList (listElement* list);

void printElement (listElement* list);

int size (listElement* list);

int indexCorrect (listElement* list, int index);

int getIndexByValue (listElement* list, int value);

listElement* addElementAtStart (listElement* list, int newValue);

listElement* addElementAtEnd (listElement* list, int newValue);

listElement* addElementAtIndex (listElement* list, int index, int newValue);

listElement* getElementByIndex (listElement* list, int index);

listElement* getElementByValue (listElement* list, int value);

listElement* removeByValue (listElement* list, int value);

listElement* removeByIndex (listElement* list, int index);

listElement* reverseList(listElement* list);