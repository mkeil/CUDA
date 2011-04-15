#include "liste.h"
#include <stdio.h>
#include <time.h>


int main(int argc, char *argv[])
{
	int elementCount;
    if (argc == 2)
    {
        elementCount = atoi(argv[1]);
    }
    else
    {
        printf ("Argumente unzulässig. 25 Elemte werden eingefüegt.\n");
        elementCount = 25;
    }


    // Aufgabe 1
    listElement *list1 = NULL;
    list1 = addElementAtStart (list1,75);
    printList (list1);
    printf ("Anzahl der Elemente %d \n",elementCount);

    // Aufgabe 2
    listElement *list = NULL;
    int i, randomNr;
    srand (time(NULL));
    listElement *tmp = NULL;
    for (i = 1; i <=elementCount; i ++)
    {
        randomNr = rand() % 10;
        // Sicherungsfunktion, falls kein Speicher mehr allokiert werden kann
        tmp = addElementAtEnd (list,randomNr);
        if (tmp == NULL)
        {
            printf ("Das %d. Elemente konnte nicht mehr eingefügt werden. Vermutlich, kein Speicher mehr vorhanden.\n",i);
            break;
        }
        else
        {
            list = tmp;
        }
    }
    
	printList(list);

    // Aufgabe 3
    list = addElementAtStart(list,4);
    list = addElementAtEnd(list,2);
    printList(list);

    // Aufgabe 4
    int middle = (int) (size(list)+1) / 2;
    
	int middleElementValue = getElementByIndex(list,middle)->value;
    printf ("Groesse der Liste: %d.\n", size(list));
    printf ("Index der Mitte ist Liste: %d.\n", middle);
    printf ("Element der Mitte ist : %d.\n", middleElementValue);
    
	printList(list);
	addElementAtIndex(list,middle+1, 300);
    printList(list);

    // Aufgabe 5
    int searchedValue = middleElementValue;
    for (i = 1; i <= size(list); i ++)
    {
        if (getElementByIndex(list,i)->value == searchedValue)
        {
            list = removeByIndex(list,i);
        }
    }
    printf ("Alle Elemente mit Wert %d wurden gelöscht.\n", middleElementValue);
    printList(list);

    // Aufgabe 6
    list = removeByIndex(list,2);
    printList(list);

    // Aufgabe 7
    list = reverseList(list);
    printList(list);

    return 0;
}