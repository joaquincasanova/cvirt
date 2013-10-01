EXEC = cvirt
SRC = cvirt.cpp
INC_DIR = /usr/include/ 
LIB_DIR = /usr/lib/
LDLIBS = -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_ml -lbcm2835
CC = g++
CFLAGS = -Wall
LFLAGS = -L$(LIB_DIR) $(LDLIBS)
IFLAGS = -I$(INC_DIR)
OBJS = $(SRC:%.cpp=%.o)

all:$(EXEC)

$(EXEC) : $(OBJS)
	$(CC) $(OBJS) $(LFLAGS) -o $(EXEC)

$(OBJS) : $(SRC)
	$(CC) $(CFLAGS) $(IFLAGS) -c $(SRC)

clean:
	rm $(EXEC) $(OBJS) *~
