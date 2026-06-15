CXX = g++
CXXFLAGS = -Wwrite-strings -Wunused-result -std=c++11 -ldl -Ofast -lpthread -ldl
INC = -I ../include
LDLIBS = -ldl -lpthread -lOpenCL
OBJS = nnlib.o


mnistTest: main.c ${OBJS}
	${CXX} ${CXXFLAGS} ${INC} -o $@ $^ ${LDLIBS}
${OBJS}: ../nnlib.c
	${CXX} -c $< ${CXXFLAGS} 
clean:
	rm -f mnistTest *.o *.out