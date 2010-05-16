# DEBUG = y
MAKERULESDIR = ..
DESTDIR = ../bin.$(UNAME)
INCLUDES = -I../include
LIBDIR = -L../lib.$(UNAME)

include $(MAKERULESDIR)/Makerules

GLSOURCES =	mesh_view.cc

ALIGNSOURCES =	mesh_align.cc \
		ICP.cc \
		KDtree.cc

SHADESOURCES =	mesh_shade.cc \
		KDtree.cc

OTHERSOURCES =	mesh_filter.cc \
		mesh_make.cc \
		mesh_cc.cc \
		mesh_cat.cc

GLOFILES = $(addprefix $(OBJDIR)/,$(GLSOURCES:.cc=.o))
ALIGNOFILES = $(addprefix $(OBJDIR)/,$(ALIGNSOURCES:.cc=.o))
SHADEOFILES = $(addprefix $(OBJDIR)/,$(SHADESOURCES:.cc=.o))
OTHEROFILES = $(addprefix $(OBJDIR)/,$(OTHERSOURCES:.cc=.o))
OFILES = $(GLOFILES) $(ALIGNOFILES) $(SHADEOFILES) $(OTHEROFILES)

GLPROGS = $(addsuffix $(EXE), $(addprefix $(DESTDIR)/, $(GLSOURCES:.cc=)))
ALIGNPROG = $(DESTDIR)/mesh_align$(EXE)
SHADEPROG = $(DESTDIR)/mesh_shade$(EXE)
OTHERPROGS = $(addsuffix $(EXE), $(addprefix $(DESTDIR)/, $(OTHERSOURCES:.cc=)))
PROGS = $(ALIGNPROG) $(SHADEPROG) $(OTHERPROGS) $(GLPROGS)

default: $(PROGS)


LIBS += -ltrimesh
$(GLPROGS) : $(DESTDIR)/%$(EXE) : $(OBJDIR)/%.o
	$(LINK) $(GLLIBS)

$(ALIGNPROG) : $(ALIGNOFILES)
	$(LINK)

$(SHADEPROG) : $(SHADEOFILES)
	$(LINK)

$(OTHERPROGS) : $(DESTDIR)/%$(EXE) : $(OBJDIR)/%.o
	$(LINK)

clean :
	-rm -f $(OFILES) $(OBJDIR)/Makedepend $(OBJDIR)/*.d
	-rm -rf $(OBJDIR)/ii_files
	-rmdir $(OBJDIR)

spotless : clean
	-rm -f $(PROGS)
	-rmdir $(DESTDIR)

