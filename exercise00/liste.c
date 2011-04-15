#include "liste.h"
#include <stdio.h>
#include <stdlib.h>


listElement* newElement (int newValue)
{
    listElement* tmp = (listElement*) malloc(sizeof(listElement));
    if (tmp != NULL)
    {
        tmp->value = newValue;
        tmp->next = NULL;
    }
    return tmp;
}

void delElement (listElement* element)
{
    free(element);
}


int indexCorrect (listElement* list, int index)
{
    if ( (index > 0) && (index <= size(list) ) )
    {
        return 1;
    }
    else
    {
        printf ("Index unzulässig! Keine Änderung vorgenommen.\n");
        return 0;
    }

}

int size (listElement* list)
{
    int a = 0;
    listElement* cursor = list;

    while (cursor != NULL)
    {
        a++;
        cursor = cursor->next;
    }
    return a;

}

void printList (listElement* list)
{
    if (size(list) == 0)
    {
        printf("Die liste ist leer.\n");
    }
    else
    {
        listElement* cursor = list;

        while (cursor != NULL)
        {
            printElement(cursor);
            cursor = cursor->next;
        }
        printf ("\n\n");
    }
}

int getIndexByValue (listElement* list, int value)
{
    if (indexCorrect)
    {
        listElement* cursor  = list;
        int found = 0;
        int index = 1;

        while (cursor->value != value && cursor->next != NULL)
        {
            cursor = cursor->next;
            index ++;
        }

        if (cursor->value == value)
        {
            found = 1;
        }

        if (!found)
        {
            index = 0;
        }
        return index;
    }
}

listElement* reverseList(listElement* list)
{
    listElement* newList = NULL;
    listElement* tmp = NULL;

    int index = size(list);
    for (index; index > 0; index --)
    {
        tmp = getElementByIndex (list, index);
        newList = addElementAtEnd (newList,tmp->value);
        list = removeByIndex (list,index);
    }
    return newList;
}

void printElement (listElement* element)
{
    printf("%d ",element->value);
}
listElement* addElementAtEnd (listElement* list, int newValue)
{
    if (size(list) == 0)
    {
        list = addElementAtStart(list, newValue);
    }
    else
    {
        listElement* newEnd = newElement(newValue);
        if (newEnd == NULL)
        {
            list = NULL;
        }
        else
        {
            listElement* oldEnd = getElementByIndex(list,size(list));
            oldEnd->next = newEnd;
        }
    }
    return list;
}


listElement* removeByIndex (listElement* list, int index)
{
    if (indexCorrect)
    {
        listElement* toDelet = NULL;
        if (index == 1)
        {
            toDelet = getElementByIndex(list, 1);
            list = list->next;
        }
        else
        {
            toDelet = getElementByIndex(list, index);
            listElement* oldPred = getElementByIndex(list, index - 1);
            oldPred->next = toDelet->next;
        }
        delElement(toDelet);
    }
    return list;
}

listElement* removeByValue (listElement* list, int value)
{
    int index = getIndexByValue (list, value);
    if (index == 0)
    {
        printf("Wert %d ist in der Liste nicht vorhanden. Keine Änderung vorgenommen\n",value);
        return NULL;
    }
    else
    {
        removeByIndex (list,index);
        return list;
    }
}



listElement* addElementAtStart (listElement* list, int newValue)
{
    listElement* newStart = newElement(newValue);

    if (newStart == NULL)
    {
        list = NULL;
    }
    else
    {
        newStart->next = list;
    }
    return newStart;
}

listElement* getElementByValue (listElement* list, int value)
{
    listElement* cursor  = list;
    int found = 0;

    while (cursor->value != value && cursor->next != NULL)
    {
        printf (" %d \n",cursor->value);
        cursor = cursor->next;
    }

    if (cursor->value == value)
    {
        found = 1;
    }

    if (found)
    {
        return cursor;
    }
    else
    {
        printf("Wert %d ist in der Liste nicht vorhanden. Keine Änderung vorgenommen\n",value);
        return NULL;
    }
}


listElement* addElementAtIndex (listElement* list, int index, int newValue)
{

    if (index == 1)
    {
        list = addElementAtStart (list, newValue);
    }
    else
    {
        listElement* newPos = newElement(newValue);
        listElement* oldPos = getElementByIndex(list,index);
        listElement* oldPred = getElementByIndex(list,index-1);
        oldPred->next = newPos;
        newPos->next = oldPos;
    }
    return list;

}



listElement* getElementByIndex (listElement* list, int index)
{
    listElement *cursor = list;
    if (indexCorrect)
    {
        int l;
        for (l = 1; l < index; l++)
        {
            cursor = cursor->next;
        }
    }
    else
    {
        return NULL;
    }
    return cursor;
}


