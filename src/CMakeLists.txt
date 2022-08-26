add_executable("Classifier" "Main.cpp")
set_property(TARGET "Classifier" PROPERTY CXX_STANDARD 17)
target_link_libraries("Classifier" PRIVATE "csv")
target_include_directories("Classifier" PRIVATE "../extern/csv-parser/include") # smh it doesn't work when I just include the library