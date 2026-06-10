run("Invert", "stack");
for (i = 0; i < 50; i++) {
	run("Dilate");
}
run("Invert", "stack");