$(function () {
    /**
     * Implements the "Predict Price" button functionality.
     */
    $('#review-form').on('click', '#submit', function (e) {
        e.preventDefault();

        // Sends the data to the app.
        // On success, updates the prediction value with the data returned by the app.
        $.ajax({
            type: "POST",
            url: "/predict",
            data: $("#review-form").serialize(),
            success: function (data) {
                $("#prediction").attr("value", "$" + data["pred"])
            }
        });

        return false;
    });
});
